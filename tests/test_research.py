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
from scripts.research_records import export_research


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
            args = ["evaluate.py", "--checkpoint", str(output / "model.pth"), "--dataset", "U1652", "--data-root", str(data),
                    "--output", str(output / "development"), "--split", "development", "--workers", "0", "--device", "cpu"]
            with patch("sample4geo.model.timm.create_model", side_effect=lambda *a, **k: TinyBackbone()), patch("sys.argv", args):
                evaluate.main()
            development = json.loads((output / "development/metrics.json").read_text())
            self.assertEqual(development["query_count"], 2)
            self.assertEqual(development["protocol"]["split"], "development")
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

    def test_campaign_all_stages_archives_scopes_resume_and_local_tables(self):
        """模拟昂贵训练命令，但真实调用指标、对比、离线报告、归档、清理和导表。"""
        from report import build_report
        from sample4geo.evaluate.metrics import score_query, summarize
        repository = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "outputs").mkdir()
            config = yaml.safe_load((repository / "configs/research_campaign.yaml").read_text())
            config.update(run="test", minimum_free_gib=0, screen_keep_top_k=0, final_seeds=[1, 2],
                arms=[{"id": "good", "module": "none"}, {"id": "bad", "module": "none"}],
                screen_tasks=[{"dataset": "U1652", "altitude": None}],
                final_tasks=[{"dataset": "U1652", "altitude": None}, {"dataset": "SUES200", "altitude": 150}])
            path = root / "campaign.yaml"
            path.write_text(yaml.safe_dump(config))
            for name in ("a", "b"):
                image = root / "images" / name / "image.png"
                image.parent.mkdir(parents=True)
                Image.new("RGB", (8, 8), "red").save(image)

            def fake_run(command, **kwargs):
                get = lambda key: command[command.index(key) + 1]
                if command[1] == "train.py":
                    output = root / "outputs" / get("--name")
                    output.mkdir()
                    (output / "model.pth").write_bytes(output.name.encode())
                    (output / "config.yaml").write_text("dataset: U1652\n")
                    (output / "train.csv").write_text("epoch,seconds,amp_skipped_steps,peak_gpu_mb\n1,1,0,20000\n")
                    with zipfile.ZipFile(output / "source.zip", "w") as archive:
                        archive.writestr("source.py", "# simulated expensive training\n")
                elif command[1] == "evaluate.py":
                    output = Path(get("--output"))
                    output.mkdir(parents=True, exist_ok=True)
                    checkpoint = Path(get("--checkpoint"))
                    correct = (True, True) if "_good_" in checkpoint.parent.name else (False, False) if "_bad_" in checkpoint.parent.name else (True, False)
                    gallery = [{"path": str(root / "images" / name / "image.png"), "key": name + "/image.png", "id": name, "included": True} for name in ("a", "b")]
                    queries = []
                    for index, match in enumerate(correct):
                        scores = [0.1, 0.1]
                        scores[index if match else 1 - index] = 0.9
                        detail = score_query(scores, ["a", "b"], gallery[index]["id"], 2)
                        candidate = lambda idx: {**gallery[idx], "score": scores[idx], "correct": idx == index}
                        detail["top_k"] = [candidate(i) for i in detail.pop("top_indices")]
                        detail["best_positive"] = candidate(detail.pop("best_positive_index"))
                        detail["best_negative"] = candidate(detail.pop("best_negative_index"))
                        queries.append({**gallery[index], **detail})
                    protocol = {"gallery_scope": get("--gallery-scope")}
                    if get("--split") == "development":
                        protocol.update(split="development", development_partition_sha256="fixed")
                    metrics = {**summarize(queries, 2), "protocol": protocol, "checkpoint_sha256": digest(checkpoint), "extra_parameters": 0}
                    atomic_json(output / "metrics.json", metrics)
                    atomic_json(output / "queries.json", {"queries": queries, "gallery": gallery,
                        "evaluation": {"dataset": "SUES200" if "sues" in checkpoint.parent.name else "U1652",
                            "altitude": 150 if "sues" in checkpoint.parent.name else None, "direction": get("--direction"), "img_size": 32}})
                elif command[1] == "report.py":
                    build_report(get("--experiment"), compare=get("--compare") if "--compare" in command else None)
                else:
                    raise AssertionError(command)
                return subprocess.CompletedProcess(command, 0)

            with patch("scripts.research_campaign.ROOT", root), patch("sample4geo.utils.snapshot_source", return_value={}):
                campaign = Campaign(path, root, 1)
                campaign.state["preflight_passed"] = True
                with patch("scripts.research_campaign.subprocess.run", side_effect=fake_run):
                    campaign.execute()
                self.assertEqual(campaign.state["status"], "complete", campaign.state)
                self.assertEqual(campaign.state["finalists"], ["good"])
                self.assertEqual(len(campaign.state["jobs"]), 13)
                self.assertEqual(len(campaign.state["results"]), 34)
                self.assertEqual(len(list((campaign.directory / "archive_ready").glob("*.json"))), 13)
                second = root / "outputs/test_final_s2_good_u1652/model.pth"
                self.assertFalse(second.exists())
                self.assertTrue((root / "outputs/test_final_s2_baseline_u1652/model.pth").exists())
                before = (campaign.directory / "commands.jsonl").read_bytes()
                resumed = Campaign(path, root, 1)
                with patch("scripts.research_campaign.subprocess.run", side_effect=AssertionError("重复训练已完成任务")):
                    resumed.execute()
                self.assertEqual(resumed.state["status"], "complete")
                self.assertEqual(before, (campaign.directory / "commands.jsonl").read_bytes())
                export_research(resumed.state, "test", root)
                self.assertTrue((root / "results/test/paired_seeds.csv").is_file())
                # 模拟SSH/SCP传输层，真实验证本地接收的哈希、解包、完整图库页面和权重策略。
                from scripts import collect_experiments as receiver
                local = root / "local"
                remote = {"state": resumed.state, "ready": [json.loads(p.read_text()) for p in sorted((campaign.directory / "archive_ready").glob("*.json"))],
                          "logs": [], "summary_files": ["metrics.csv", "protocol.yaml"]}
                polls = []

                def fake_ssh(*args, **kwargs):
                    polls.append(args)
                    return json.dumps(remote) if len(polls) == 1 else "acknowledged"

                def fake_scp(command, **kwargs):
                    self.assertEqual(command[0], "scp")
                    source = command[-2].split(":", 1)[1]
                    prefix = "/root/autodl-tmp/CVGL_NEW/outputs/test/"
                    source = campaign.directory / source[len(prefix):] if source.startswith(prefix) else Path(source)
                    shutil.copy2(source, command[-1])
                    return subprocess.CompletedProcess(command, 0)

                with patch.object(receiver, "ROOT", local), patch.object(receiver, "load_policy", return_value=None), \
                     patch.object(receiver.subprocess, "check_output", side_effect=fake_ssh), \
                     patch.object(receiver.subprocess, "run", side_effect=fake_scp), \
                     patch.object(receiver.time, "sleep", side_effect=AssertionError("接收没有在终态结束")):
                    receiver.main(["--host", "root@fake", "--port", "22", "--socket", "/tmp/fake", "--run", "test"])
                artifacts = local / "outputs/test/artifacts"
                self.assertTrue((artifacts / "final_s1_good_sues200_150/all/D2S/report/index.html").is_file())
                self.assertTrue((artifacts / "final_s1_good_u1652/model.pth").is_file())
                self.assertFalse((artifacts / "final_s2_good_u1652/model.pth").exists())
                manifest = json.loads((local / "outputs/test/archive_manifest.json").read_text())
                self.assertEqual(manifest["status"], "complete")
                self.assertEqual(len(manifest["artifacts"]), 13)
                item = manifest["artifacts"]["final_s1_good_u1652"]
                self.assertEqual(item["checkpoint_retention_policy_sha256"], resumed.state["config_sha256"])
                self.assertEqual(item["checkpoint_retention_code_sha256"], resumed.state["code_sha256"])


if __name__ == "__main__":
    unittest.main()
