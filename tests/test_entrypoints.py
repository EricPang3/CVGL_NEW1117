"""用小编码器贯通真实入口；删除训练图片后独立评估。"""

import json
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image
from torch import nn

import evaluate
import train
from report import build_report
from sample4geo.utils import ROOT, read_config


class SmallEncoder(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.model = nn.Linear(3, 8)
        self.logit_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, image, other=None):
        encode = lambda value: self.model(value.mean(dim=(2, 3)))
        return encode(image) if other is None else (encode(image), encode(other))

    def data_config(self):
        return {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}


class EntrypointTests(unittest.TestCase):
    def test_train_independent_eval_and_offline_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, output = root / "data", root / "experiment"
            output.mkdir()
            for label in range(4):
                for view in ("train/satellite", "train/drone", "test/query_drone", "test/gallery_satellite"):
                    path = data / view / str(label) / "image.png"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    Image.new("RGB", (32, 32), (label * 50, 100, 200)).save(path)
            argv = ["train.py", "--config", str(ROOT / "configs/u1652.yaml"), "--data-root", str(data),
                    "--batch-size", "2", "--img-size", "32", "--workers", "0", "--device", "cpu", "--no-pretrained"]
            with patch("sys.argv", argv), patch.object(train, "TimmModel", SmallEncoder), patch.object(train, "new_output", return_value=output):
                train.main()
            config = read_config(output / "config.yaml")
            self.assertEqual(config["epochs"], 1)
            self.assertEqual(config["amp_init_scale"], 1024)
            self.assertEqual(config["planned_steps_per_epoch"], 2)
            self.assertEqual(config["sampler_rng"], "global_python")
            self.assertTrue((output / "source.zip").is_file())
            shutil.rmtree(data / "train")
            evaluation = output / "D2S"
            argv = ["evaluate.py", "--checkpoint", str(output / "model.pth"), "--output", str(evaluation), "--device", "cpu"]
            with patch("sys.argv", argv), patch.object(evaluate, "TimmModel", SmallEncoder):
                evaluate.main()
            metrics = json.loads((evaluation / "metrics.json").read_text())
            queries = json.loads((evaluation / "queries.json").read_text())
            self.assertEqual(metrics["query_count"], 4)
            self.assertEqual(metrics["protocol"]["gallery_scope"], "query_ids")
            self.assertEqual(queries["queries"][0]["key"], "test/query_drone/0/image.png")
            self.assertIn("best_negative", queries["queries"][0])
            build_report(evaluation)
            with zipfile.ZipFile(evaluation / "report.zip") as archive:
                self.assertIn("training_config.yaml", archive.namelist())
                self.assertIn("source.zip", archive.namelist())
                self.assertIn("train.csv", archive.namelist())
                self.assertNotIn("model.pth", archive.namelist())
