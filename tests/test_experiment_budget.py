"""验证预算只影响新任务、同seed对照及完整归档门槛。"""

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import yaml

from scripts.collect_overnight import verified_manifest
from scripts.experiment_budget import budget_expired, parse_deadline
from scripts.run_overnight import repeat_config
from scripts import module_suite


class BudgetTests(unittest.TestCase):
    def test_timezone_and_boundary(self):
        deadline = parse_deadline("2026-11-21T12:00:00+08:00")
        self.assertFalse(budget_expired(deadline, datetime(2026, 11, 21, 3, 59, tzinfo=timezone.utc)))
        self.assertTrue(budget_expired(deadline, datetime(2026, 11, 21, 4, 0, tzinfo=timezone.utc)))
        self.assertFalse(budget_expired(None))
        with self.assertRaises(ValueError):
            parse_deadline("2026-11-21T12:00:00")

    def test_repeat_preserves_recipe_and_fixed_order(self):
        base = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/module_suite.yaml").read_text())
        original = copy.deepcopy(base)
        repeated = repeat_config(base, 2, parse_deadline("2026-11-21T12:00:00+08:00"))
        self.assertEqual(base, original)
        for field in ("modules", "tasks", "directions", "epochs", "batch_size", "gallery_scope"):
            self.assertEqual(repeated[field], original[field])
        self.assertEqual(repeated["seed"], 2)
        self.assertNotEqual(repeated["baseline_run"], original["baseline_run"])
        self.assertIn("seed2", repeated["baseline_run"])

    def test_archive_gate(self):
        complete = {"status": "complete", "expected_archive_tasks": 2, "artifacts": {"a": {}, "b": {}}}
        self.assertTrue(verified_manifest(complete))
        self.assertFalse(verified_manifest({**complete, "artifacts": {"a": {}}}))
        self.assertFalse(verified_manifest({**complete, "status": "partial_failure"}))
        self.assertTrue(verified_manifest({**complete, "status": "budget_reached"}))

    def test_running_job_finishes_after_deadline_and_next_job_stays_pending(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = {"run": "repeat", "baseline_run": "seed2_baseline", "seed": 2, "epochs": 1,
                      "batch_size": 24, "directions": ["D2S", "S2D"], "gallery_scope": "query_ids",
                      "top_k": 10, "report_limit": 2, "bootstrap_baseline": True,
                      "stop_starting_at": "2026-11-21T12:00:00+08:00",
                      "tasks": [{"dataset": "U1652", "altitude": None}],
                      "modules": [{"id": "eca", "kind": "train", "alpha": .1},
                                  {"id": "wtconv", "kind": "train", "alpha": .1}]}
            path = root / "config.yaml"
            path.write_text(yaml.safe_dump(config))
            out = root / "outputs/repeat"
            out.mkdir(parents=True)
            (out / "gpu_preflight.json").write_text('{"passed": true}')
            expired = False
            calls = []

            def run(command, **kwargs):
                nonlocal expired
                args = command[1:]
                calls.append(args)
                if args[0] == "train.py":
                    self.assertEqual(args[args.index("--seed") + 1], "2")
                    model = root / "outputs" / args[args.index("--name") + 1]
                    model.mkdir()
                    (model / "model.pth").write_bytes(b"test checkpoint")
                    if args[args.index("--module") + 1] == "eca":
                        expired = True  # 训练开始后越过时间边界。
                elif args[0] == "evaluate.py":
                    output = Path(args[args.index("--output") + 1])
                    output.mkdir(parents=True)
                    metric = {"recall@1": .5, "recall@5": .7, "recall@10": .9, "mAP": .6,
                              "checkpoint_sha256": hashlib.sha256(b"test checkpoint").hexdigest()}
                    (output / "metrics.json").write_text(json.dumps(metric))
                elif args[0] == "report.py":
                    report = Path(args[args.index("--experiment") + 1]) / "report"
                    report.mkdir()
                    (report / "comparison.json").write_text('{"counts":{"fixed":0,"regressed":0}}')
                else:
                    self.fail(str(args))

            with patch.object(module_suite, "ROOT", root), patch.object(module_suite.subprocess, "run", side_effect=run), \
                    patch.object(module_suite, "budget_expired", side_effect=lambda _: expired), \
                    patch.object(sys, "argv", ["module_suite.py", "--config", str(path)]):
                module_suite.main()
            state = json.loads((out / "state.json").read_text())
            self.assertEqual(state["status"], "budget_reached")
            self.assertEqual(state["expected_archive_tasks"], 2)
            self.assertEqual(state["planned_archive_tasks"], 3)
            self.assertEqual(state["jobs"]["eca_u1652"]["status"], "complete")
            self.assertNotIn("status", state["jobs"]["wtconv_u1652"])
            self.assertEqual(len(state["results"]), 2)
            self.assertEqual(len(list((out / "archive_ready").glob("*.json"))), 2)
            self.assertEqual(sum(c[0] == "train.py" for c in calls), 2)
            compare = [c for c in calls if "--compare" in c]
            self.assertEqual(len(compare), 2)
            self.assertTrue(all("seed2_baseline_u1652" in c[c.index("--compare") + 1] for c in compare))


if __name__ == "__main__":
    unittest.main()
