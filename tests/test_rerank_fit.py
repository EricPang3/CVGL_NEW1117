"""小编码器贯通训练地点拟合→序列化→独立重排评估，不冒充正式模型成绩。"""

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

from scripts.rerank_experiment import fit, evaluate_reranker


class SpatialEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
                                    nn.Conv2d(64, 1024, 1), nn.ReLU(), nn.AdaptiveAvgPool2d((12, 12)))

    def forward_features(self, x):
        return self.layers(x)

    def forward_head(self, x):
        return x.mean((2, 3))


class RerankFitTests(unittest.TestCase):
    def test_all_rerankers_fit_training_only_and_evaluate_unseen_ids(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                rng = np.random.default_rng(1)
                for place in range(6):
                    views = ("train/satellite", "train/drone") if place < 4 else ("test/query_drone", "test/gallery_satellite")
                    for view in views:
                        for i in range(2 if view.endswith("drone") else 1):
                            path = root / "data" / view / str(place) / f"{i}.png"
                            path.parent.mkdir(parents=True, exist_ok=True)
                            Image.fromarray(rng.integers(0, 256, (32,32,3), dtype=np.uint8)).save(path)
                config = {"dataset":"U1652", "data_root":str(root/"data"), "img_size":32,
                          "mean":[.485,.456,.406], "std":[.229,.224,.225], "eval_batch_size":4,
                          "workers":0, "amp":False, "direction":"D2S", "altitude":None,
                          "gallery_scope":"query_ids", "top_k":2, "module":"none"}
                torch.manual_seed(1)
                model = nn.Module()
                model.model = SpatialEncoder()
                model.eval()
                checkpoint = root / "baseline.pth"
                torch.save({"model":model.state_dict(),"config":config}, checkpoint)
                for method in ("patch_netvlad", "r2former_rerank", "superglobal_rerank"):
                    with self.subTest(method=method):
                        output = root / method
                        args = Namespace(output=str(output), checkpoint=str(checkpoint), method=method,
                                         seed=1, head=str(output/"reranker.pth"), candidates=20)
                        fit(args, model, config, torch.device("cpu"))
                        fitted = torch.load(args.head, map_location="cpu", weights_only=True)
                        self.assertEqual(fitted["metadata"]["fit_scope"], "official_training_places_only")
                        if method != "superglobal_rerank":
                            self.assertEqual(fitted["metadata"]["training_ids"], ["0","1","2","3"])
                        args.output = str(output / "D2S")
                        evaluate_reranker(args, model, dict(config), torch.device("cpu"))
                        result = json.loads((Path(args.output)/"metrics.json").read_text())
                        queries = json.loads((Path(args.output)/"queries.json").read_text())
                        self.assertTrue(0 <= result["recall@1"] <= 1)
                        self.assertEqual(result["reranking"]["candidates"], 2)
                        self.assertEqual({q["id"] for q in queries["queries"]}, {"4","5"})
                        self.assertEqual(result["protocol"]["ranking"], method)
        finally:
            torch.set_num_threads(threads)
