"""一轮配方的受控消融：真实权重轨迹、参数匹配、同seed配对。"""

import csv
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

import evaluate
import train
from sample4geo.model import TimmModel, load_model_state
from sample4geo.modules import make_module
from sample4geo.trainer import cosine_scheduler
from sample4geo.utils import configure, file_hash, read_config
from scripts.research_records import export_research
from test_research import TinyBackbone, dataset_fixture


class AblationTests(unittest.TestCase):
    def test_schedule_horizon_keeps_default_and_rejects_wrapping(self):
        def rates(epochs, horizon=None):
            optimizer = torch.optim.SGD([nn.Parameter(torch.zeros(1))], lr=0.001)
            config = {"epochs": epochs, "warmup_epochs": 0.1}
            if horizon is not None:
                config["schedule_epochs"] = horizon
            scheduler = cosine_scheduler(optimizer, 10, config)
            values = []
            for _ in range(epochs * 10):
                values.append(optimizer.param_groups[0]["lr"])
                optimizer.step()
                scheduler.step()
            return values
        self.assertEqual(rates(1), rates(1, 1))
        self.assertEqual(rates(1, 3), rates(3, 3)[:10])
        self.assertNotEqual(rates(1), rates(1, 3))
        root = Path(__file__).resolve().parents[1]
        config = read_config(root / "configs/u1652.yaml")
        for horizon in (2, float("nan"), float("inf")):
            with self.subTest(horizon=horizon), self.assertRaisesRegex(ValueError, "schedule_epochs"):
                configure({**config, "epochs": 3, "schedule_epochs": horizon}, Namespace())
        for snapshots in ([0], [4], [True], "1"):
            with self.subTest(snapshots=snapshots), self.assertRaisesRegex(ValueError, "checkpoint_epochs"):
                configure({**config, "epochs": 3, "checkpoint_epochs": snapshots}, Namespace())

    def test_matched_control_parameters_gradients_and_zero_branch(self):
        wt = make_module("wtconv")[0]
        dw = make_module("dwconv_matched")[0]
        count = lambda m: sum(p.numel() for p in m.parameters() if p.requires_grad)
        self.assertEqual(count(wt), 52224)
        self.assertEqual(count(wt), count(dw))
        x = torch.randn(2, 1024, 12, 12, requires_grad=True)
        dw(x).square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in dw.parameters()))
        with patch("sample4geo.model.timm.create_model", side_effect=lambda *a, **k: TinyBackbone()):
            baseline, candidate = TimmModel().eval(), TimmModel(module="dwconv_matched", module_alpha=0).eval()
            load_model_state(candidate, baseline.state_dict(), initialize_module=True)
            image = torch.randn(2, 3, 8, 8)
            self.assertTrue(torch.equal(baseline(image), candidate(image)))

    def test_real_three_epoch_prefix_is_immutable_and_independently_evaluable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            dataset_fixture(data)
            prefix_hash = []
            save = train.save_checkpoint
            def traced_save(model, config, epoch, output):
                save(model, config, epoch, output)
                if epoch == 1:
                    prefix_hash.append(file_hash(output / "model_epoch1.pth"))
            for epochs, name in ((3, "long"), (1, "prefix")):
                output = root / name
                output.mkdir()
                args = ["train.py", "--dataset", "U1652", "--data-root", str(data), "--batch-size", "2",
                        "--img-size", "32", "--workers", "0", "--device", "cpu", "--no-pretrained",
                        "--development-fraction", "0.25", "--epochs", str(epochs), "--schedule-epochs", "3", "--checkpoint-epochs", "1"]
                with patch("sample4geo.model.timm.create_model", side_effect=lambda *a, **k: TinyBackbone()), patch.object(train, "new_output", return_value=output), patch.object(train, "save_checkpoint", side_effect=traced_save), patch("sys.argv", args):
                    train.main()
            long = root / "long"
            self.assertEqual(file_hash(long / "model_epoch1.pth"), prefix_hash[0])
            prefix = torch.load(long / "model_epoch1.pth", weights_only=True)
            last = torch.load(long / "model.pth", weights_only=True)
            truncated = torch.load(root / "prefix/model.pth", weights_only=True)
            self.assertEqual((prefix["epoch"], prefix["config"]["checkpoint_epoch"], last["epoch"]), (1, 1, 3))
            self.assertTrue(all(torch.equal(v, truncated["model"][k]) for k, v in prefix["model"].items()))
            self.assertTrue(any(not torch.equal(v, last["model"][k]) for k, v in prefix["model"].items()))
            self.assertEqual(json.loads((long / "checkpoint_epochs.json").read_text())["1"]["sha256"], prefix_hash[0])
            args = ["evaluate.py", "--checkpoint", str(long / "model_epoch1.pth"), "--output", str(long / "development"), "--split", "development", "--device", "cpu", "--workers", "0"]
            with patch("sample4geo.model.timm.create_model", side_effect=lambda *a, **k: TinyBackbone()), patch("sys.argv", args):
                evaluate.main()
            metrics = json.loads((long / "development/metrics.json").read_text())
            self.assertEqual((metrics["checkpoint_epoch"], metrics["query_count"]), (1, 2))
            before = file_hash(long / "model.pth")
            with self.assertRaisesRegex(RuntimeError, "拒绝覆盖"):
                save(nn.Linear(1, 1), {"checkpoint_epochs": [1]}, 1, long)
            self.assertEqual(file_hash(long / "model.pth"), before)

    def test_ablation_table_pairs_seed_scope_and_rejects_mixed_recipe(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = []
            for seed in (1, 2, 3):
                for scope in ("query_ids", "all"):
                    for variant in ("baseline", "consensus", "wt_consensus", "dw_consensus"):
                        rows.append({"artifact_job": f"s{seed}_{variant}", "phase": "final", "split": "test", "variant": variant,
                            "seed": seed, "dataset": "SUES200", "altitude": 150, "direction": "D2S", "gallery_scope": scope,
                            "recall@1": 90 + (seed if variant == "wt_consensus" else 1 if variant == "dw_consensus" else 0),
                            "mAP": 90, "delta_R1_pp": 0, "delta_AP_pp": 0, "checkpoint_sha256": f"s{seed}_{variant}",
                            "epochs": 1, "schedule_epochs": 1, "batch_size": 24, "img_size": 384})
            state = {"selection_mode": "fixed_ablation", "results": rows, "final_seeds": [1, 2, 3],
                     "jobs": {r["artifact_job"]: {"status": "complete"} for r in rows}}
            export_research(state, "test", tmp)
            with (Path(tmp) / "results/test/ablation_summary.csv").open(encoding="utf-8-sig") as stream:
                summary = list(csv.DictReader(stream))
            matched = [r for r in summary if r["candidate"] == "wt_consensus" and r["control"] == "dw_consensus"]
            self.assertEqual(len(matched), 2)
            self.assertTrue(all(r["n"] == "3" and float(r["mean_delta_R1_pp"]) == 1 for r in matched))
            rows[-1]["epochs"] = 3
            with self.assertRaisesRegex(ValueError, "配方不一致"):
                export_research(state, "test", tmp)
