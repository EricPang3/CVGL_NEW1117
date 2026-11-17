"""无需下载权重或使用真实数据的关键语义检查。"""

import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from argparse import Namespace
from unittest.mock import patch

import torch
from PIL import Image

from report import build_report
from sample4geo.dataset.university import UniqueBatchSampler, build_records
from sample4geo.evaluate.metrics import score_query, summarize
from sample4geo.loss import InfoNCE
from sample4geo.utils import ROOT, configure, read_config


class CoreTests(unittest.TestCase):
    def test_data_root_priority_for_local_and_autodl(self):
        args = Namespace(data_root=None)
        with patch.dict(os.environ, {}, clear=True):
            config = configure(read_config(ROOT / "configs/sues200.yaml"), args)
            self.assertEqual(Path(config["data_root"]), Path.home() / "datasets/SUES200")
        with patch.dict(os.environ, {"CVGL_DATA_ROOT": "/tmp/autodl-datasets"}):
            config = configure(read_config(ROOT / "configs/sues200.yaml"), args)
            self.assertEqual(config["data_root"], "/tmp/autodl-datasets/SUES200")
            args.data_root = "/tmp/explicit-sues"
            config = configure(read_config(ROOT / "configs/sues200.yaml"), args)
            self.assertEqual(config["data_root"], "/tmp/explicit-sues")

    def test_ranking_multiple_positives_and_missing(self):
        detail = score_query([0.9, 0.8, 0.7, 0.6], ["b", "a", "c", "a"], "a", 3)
        self.assertEqual(detail["first_positive_rank"], 2)
        self.assertEqual(detail["positive_count"], 2)
        self.assertEqual(detail["top_indices"], [0, 1, 2])
        self.assertAlmostEqual(detail["ap"], 1 / 3)
        missing = score_query([0.9], ["b"], "a", 10)
        metrics = summarize([detail, missing], 4)
        self.assertEqual(metrics["missing_positive_queries"], 1)
        self.assertEqual(metrics["recall@1"], 0)
        self.assertEqual(metrics["recall@5"], 1)
        self.assertEqual(metrics["recall@1%"], 0)
        with self.assertRaises(ValueError):
            summarize([missing], 1)

    def test_tie_order_and_one_percent_boundary(self):
        detail = score_query([1, 1], ["a", "b"], "b", 10)
        self.assertEqual(detail["first_positive_rank"], 2)
        self.assertEqual(summarize([detail], 100)["recall@1%"], 0)
        self.assertEqual(summarize([detail], 101)["recall@1%"], 1)

    def test_sampler_has_unique_places_and_no_repeated_pairs(self):
        pairs = [(str(i % 7), "sat", "drone") for i in range(53)]
        sampler = UniqueBatchSampler(pairs, 4, 9)
        flattened = [i for batch in sampler for i in batch]
        self.assertEqual(len(flattened), len(set(flattened)))
        for batch in sampler:
            self.assertEqual(len({pairs[i][0] for i in batch}), 4)
        self.assertEqual(len(flattened) + sampler.dropped, 53)
        self.assertEqual(sampler.batches, UniqueBatchSampler(pairs, 4, 9).batches)
        with self.assertRaises(ValueError):
            UniqueBatchSampler(pairs, 8, 1)

    def test_loss_backward_and_view_symmetry(self):
        torch.manual_seed(1)
        sat, drone = torch.randn(4, 8, requires_grad=True), torch.randn(4, 8, requires_grad=True)
        criterion = InfoNCE()
        loss = criterion(sat, drone, torch.tensor(14.0))
        self.assertTrue(torch.allclose(loss, criterion(drone, sat, torch.tensor(14.0))))
        loss.backward()
        self.assertTrue(torch.isfinite(sat.grad).all())
        self.assertGreater(sat.grad.abs().sum().item(), 0)

    def test_raw_sues_split_and_both_directions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index in range(1, 201):
                place = f"{index:04d}"
                for view in (root / "satellite-view" / place, root / "drone_view_512" / place / "150"):
                    view.mkdir(parents=True)
                    Image.new("RGB", (8, 8)).save(view / "image.png")
            config = read_config(ROOT / "configs/sues200.yaml")
            config["data_root"] = tmp
            sat, drone = build_records(config, training=True)
            self.assertEqual((len(sat), len(drone)), (120, 120))
            for direction in ("D2S", "S2D"):
                config["direction"] = direction
                query, gallery = build_records(config)
                self.assertEqual((len(query), len(gallery)), (80, 200))
                self.assertFalse({r["id"] for r in query} & {r["id"] for r in sat})

    def test_offline_report_contains_images_and_safe_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "source.png"
            Image.new("RGB", (8, 8), "red").save(image)
            candidate = {"path": str(image), "id": "a", "score": 1.0, "correct": True}
            query = {"path": str(image), "id": "</script>", "valid": True,
                     "ap": 1.0, "first_positive_rank": 1, "top1_correct": True,
                     "best_positive": candidate, "top_k": [candidate]}
            metrics = {"mAP": 1, **{f"recall@{k}": 1 for k in ("1", "5", "10", "1%")},
                       "gallery_count": 1, "protocol": {"gallery_scope": "all"}}
            (root / "queries.json").write_text(json.dumps({"queries": [query]}))
            (root / "metrics.json").write_text(json.dumps(metrics))
            report = build_report(root)
            html = (report / "index.html").read_text()
            self.assertNotIn('"id": "</script>"', html)
            self.assertNotIn("fetch(", html)
            with zipfile.ZipFile(root / "report.zip") as archive:
                self.assertIn("index.html", archive.namelist())
                self.assertEqual(len([n for n in archive.namelist() if n.endswith(".jpg")]), 1)


if __name__ == "__main__":
    unittest.main()
