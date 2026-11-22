"""新研究流程的边界：开发集无地点泄漏、训练期辅助独立推理、只凭开发集晋级、完整归档后清理。"""

import json
import random
import shutil
import subprocess
import tempfile
import unittest
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import torch
import yaml
from PIL import Image
from torch import nn

import evaluate
import train
from sample4geo.dataset.development import development_partition
from sample4geo.dataset.university import build_records
from sample4geo.loss import InfoNCE
from sample4geo.model import TimmModel, load_model_state
from sample4geo.modules import make_module
from sample4geo.research_modules import FEATURE_MODULES, FrequencyGate
from scripts.research_campaign import BudgetReached, Campaign, assess, atomic_json, digest, job_name


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_features = 8
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.norm = nn.LayerNorm(8)
        self.pretrained_cfg = {"mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225), "input_size": (3, 32, 32)}

    def forward_features(self, x):
        return self.conv(x)

    def forward_head(self, x):
        return self.norm(x.mean((2, 3)))

    def forward(self, x):
        return self.forward_head(self.forward_features(x))


def dataset_fixture(root, places=8):
    for label in range(places):
        for view in ("train/satellite", "train/drone", "test/query_drone", "test/gallery_satellite"):
            image = root / view / str(label) / "image.png"
            image.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (32, 32), (label * 25, 80, 180)).save(image)


class ResearchTests(unittest.TestCase):
    def test_development_places_disjoint_stable_and_does_not_consume_rng(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_fixture(root)
            config = {"dataset": "U1652", "data_root": str(root), "direction": "D2S",
                      "development_fraction": 0.25, "development_seed": 10}
            before = random.getstate()
            partition = development_partition(config)
            self.assertEqual(before, random.getstate())
            self.assertEqual(partition, development_partition({**config, "seed": 999}))
            sat, _ = build_records(config, training=True)
            queries, gallery = build_records({**config, "eval_split": "development"})
            self.assertEqual(len(sat), 6)
            self.assertEqual(len(queries), 2)
            self.assertFalse({r["id"] for r in sat} & {r["id"] for r in queries})
            self.assertEqual({r["id"] for r in queries}, {r["id"] for r in gallery})
            shutil.rmtree(root / "train")
            official, _ = build_records({**config, "eval_split": "test"})
            self.assertEqual(len(official), 8)

    def test_new_branches_finite_real_map_and_frequency_survives_gap(self):
        for name in FEATURE_MODULES:
            with self.subTest(name=name):
                module, mode, dimension = make_module(name)
                value = (torch.randn(2, 1024, 12, 12) * 32).requires_grad_()
                output = module(value)
                self.assertEqual(output.shape, (2, dimension) if mode == "parallel" else value.shape)
                self.assertTrue(torch.isfinite(output).all())
                output.square().mean().backward()
                self.assertTrue(torch.isfinite(value.grad).all())
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in module.parameters() if p.grad is not None))
        module = FrequencyGate(8)
        module(torch.randn(2, 8, 12, 12)).mean((2, 3)).square().mean().backward()
        self.assertGreater(module.cutoff.grad.abs().item(), 0)
        self.assertGreater(module.gains.grad[1].abs().sum().item(), 0)

    def test_auxiliary_gradients_and_pair_free_inference(self):
        for name in ("consensus", "visible_consensus", "part_align", "view_relation", "vicreg"):
            with self.subTest(name=name), patch("sample4geo.model.timm.create_model", side_effect=lambda *a, **k: TinyBackbone()):
                model = TimmModel(auxiliary=name)
                a, b = torch.randn(4, 3, 8, 8), torch.randn(4, 3, 8, 8)
                loss, _, auxiliary = model.training_loss(a, b, InfoNCE(), 0.1)
                self.assertTrue(torch.isfinite(auxiliary))
                loss.backward()
                self.assertGreater(model.model.conv.weight.grad.abs().sum().item(), 0)
                model.eval()
                with patch.object(model.training_head, "forward", side_effect=AssertionError("推理调用了成对辅助分支")):
                    single = model(a)
                    pair, _ = model(a, b * 10)
                    self.assertTrue(torch.equal(single, pair))

    def test_zero_research_branch_matches_old_state(self):
        with patch("sample4geo.model.timm.create_model", side_effect=lambda *a, **k: TinyBackbone()):
            baseline = TimmModel().eval()
            image = torch.randn(2, 3, 8, 8)
            self.assertTrue(torch.equal(baseline(image), baseline.model(image)))
            for name in FEATURE_MODULES:
                candidate = TimmModel(module=name, module_alpha=0).eval()
                load_model_state(candidate, baseline.state_dict(), initialize_module=True)
                self.assertTrue(torch.equal(candidate(image), baseline(image)), name)

    def test_real_train_auxiliary_then_independent_evaluation_without_training_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, output = root / "data", root / "experiment"
            output.mkdir()
            dataset_fixture(data)
            args = ["train.py", "--dataset", "U1652", "--data-root", str(data), "--batch-size", "2",
                    "--img-size", "32", "--workers", "0", "--device", "cpu", "--no-pretrained",
                    "--module", "prototype_pool", "--auxiliary", "consensus", "--development-fraction", "0.25"]
            with patch("sample4geo.model.timm.create_model", side_effect=lambda *a, **k: TinyBackbone()), patch.object(train, "new_output", return_value=output), patch("sys.argv", args):
                train.main()
            # 已参与完整训练的旧权重不得被当作同地点开发集权重使用。
            weights = torch.load(output / "model.pth", weights_only=True)
            weights["config"]["training_ids"] += weights["config"]["development_partition"]["development_ids"]
            torch.save(weights, output / "leaky.pth")
            args = ["evaluate.py", "--checkpoint", str(output / "leaky.pth"), "--output", str(output / "leaky"), "--split", "development", "--device", "cpu"]
            with patch("sample4geo.model.timm.create_model", side_effect=lambda *a, **k: TinyBackbone()), patch("sys.argv", args), self.assertRaises(SystemExit):
                evaluate.main()
            shutil.rmtree(data / "train")
            args = ["evaluate.py", "--checkpoint", str(output / "model.pth"), "--dataset", "U1652", "--data-root", str(data),
                    "--output", str(output / "D2S"), "--workers", "0", "--device", "cpu", "--batch-size", "2"]
            with patch("sample4geo.model.timm.create_model", side_effect=lambda *a, **k: TinyBackbone()), patch("sys.argv", args):
                evaluate.main()
            metrics = json.loads((output / "D2S/metrics.json").read_text())
            self.assertEqual(metrics["query_count"], 8)
            self.assertGreater(metrics["training_only_parameters"], 0)
            self.assertNotIn("split", metrics["protocol"])

    def test_selection_ignores_test_scores_requires_all_tasks_and_paired_positive(self):
        tasks = [{"dataset": "U1652", "altitude": None}, {"dataset": "SUES200", "altitude": 150}]
        rows = []
        for phase in ("screen", "confirm"):
            for task in tasks:
                for direction in ("D2S", "S2D"):
                    for arm in ("baseline", "good", "lucky", "bad"):
                        ap = 90 + (0.4 if arm == "good" else 0.8 if arm == "lucky" and phase == "screen" else -0.1 if arm == "lucky" else 0)
                        rows.append({"phase": phase, **task, "variant": arm, "direction": direction, "mAP": ap,
                                     "recall@1": 90 - (2 if arm == "bad" else 0), "split": "development", "gallery_scope": "query_ids"})
        rows += [{**r, "mAP": 100 if r["variant"] == "bad" else 0, "split": "test"} for r in rows]
        assessed = assess(rows, ["good", "lucky", "bad", "missing"], ["screen", "confirm"], tasks, 0.2, 1)
        self.assertEqual([r["variant"] for r in assessed if r["eligible"]], ["good"])

    def test_archive_before_deletion_protects_baseline_and_records_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "outputs").mkdir()
            config = {"run": "test", "arms": [{"id": "a", "module": "none"}], "minimum_free_gib": 0}
            path = root / "config.yaml"
            path.write_text(yaml.safe_dump(config))
            with patch("scripts.research_campaign.ROOT", root), patch("sample4geo.utils.snapshot_source", return_value={}):
                campaign = Campaign(path, root, 1)
                model = root / "outputs/test_screen_s1_a_u1652"
                model.mkdir()
                (model / "model.pth").write_bytes(b"weights")
                for name in ("config.yaml", "source.zip", "train.csv"):
                    (model / name).write_bytes(b"record")
                for direction in ("D2S", "S2D"):
                    (model / direction).mkdir()
                    for name in ("metrics.json", "queries.json"):
                        (model / direction / name).write_text("{}")
                job = "screen_s1_a_u1652"
                campaign.state["jobs"][job] = {"phase": "screen", "seed": 1, "variant": "a", "status": "complete", "model_dir": str(model), "checkpoint_sha256": digest(model / "model.pth")}
                with self.assertRaisesRegex(RuntimeError, "未归档"):
                    campaign.prune(job)
                campaign.archive(job)
                campaign.state["jobs"][job]["variant"] = "baseline"
                campaign.prune(job)
                self.assertTrue((model / "model.pth").exists())
                campaign.state["jobs"][job]["variant"] = "a"
                campaign.prune(job)
                self.assertFalse((model / "model.pth").exists())
                self.assertEqual(json.loads((model / "WEIGHT_DELETED.json").read_text())["status"], "deleted")
                ready = json.loads((campaign.directory / "archive_ready" / f"{job}.json").read_text())
                self.assertEqual(ready["bundle_sha256"], digest(ready["bundle"]))
                self.assertEqual(ready["checkpoint_storage"], "discarded_on_server_after_records")
                campaign.state["deadline"] = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
                with self.assertRaises(BudgetReached):
                    campaign.boundary()
                campaign.save()
                with self.assertRaisesRegex(ValueError, "预算改变"):
                    Campaign(path, root, 2)


if __name__ == "__main__":
    unittest.main()
