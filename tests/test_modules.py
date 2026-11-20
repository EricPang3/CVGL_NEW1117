"""模块机制边界：真实1024×12×12、梯度、旧权重、关闭分支和重排候选范围。"""

import unittest
from unittest.mock import patch

import numpy as np
import torch

from sample4geo.model import TimmModel, load_model_state
from sample4geo.module_adapters import component
from sample4geo.modules import make_module
from sample4geo.utils import ROOT, read_config


class ModuleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_catalog_has_33_independent_modules_and_controls(self):
        config = read_config(ROOT / "configs/module_suite.yaml")
        main = [m for m in config["modules"] if "control_for" not in m]
        self.assertEqual(len(main), 33)
        self.assertEqual(len({m["id"] for m in main}), 33)
        self.assertEqual(config["batch_size"], 24)
        self.assertEqual(len(config["tasks"]), 5)
        self.assertEqual(config["directions"], ["D2S", "S2D"])

    def test_all_feature_modules_shape_finite_and_backward(self):
        config = read_config(ROOT / "configs/module_suite.yaml")
        for entry in config["modules"]:
            if entry["kind"] == "rerank":
                continue
            with self.subTest(module=entry["id"]):
                torch.manual_seed(1)
                module, mode, dimension = make_module(entry["id"])
                inputs = torch.randn(2, 1024, 12, 12, requires_grad=True)
                output = module(inputs)
                output = output[0] if isinstance(output, tuple) else output
                self.assertEqual(tuple(output.shape), (2, dimension) if mode == "parallel" else (2, 1024, 12, 12))
                self.assertTrue(torch.isfinite(output).all())
                output.float().square().mean().backward()
                self.assertIsNotNone(inputs.grad)
                self.assertTrue(torch.isfinite(inputs.grad).all())

    def test_none_legacy_and_zero_branch_equal_original_encoder(self):
        torch.manual_seed(1)
        reference = TimmModel().eval()
        inputs = torch.randn(1, 3, 384, 384)
        with torch.no_grad():
            expected = reference.model(inputs)
            self.assertTrue(torch.equal(reference(inputs), expected))
            features = reference.model.forward_features(inputs)
            self.assertEqual(tuple(features.shape), (1, 1024, 12, 12))
        for name in ("eca", "wtconv", "boq", "burst_gap", "grn", "salad", "gem"):
            with self.subTest(module=name), patch("sample4geo.model.timm.create_model", return_value=reference.model):
                candidate = TimmModel(module=name, module_alpha=0).eval()
                missing = load_model_state(candidate, reference.state_dict(), initialize_module=True)
                self.assertTrue(missing)
                with torch.no_grad():
                    self.assertTrue(torch.equal(candidate.encode_map(features), expected))
        self.assertEqual(set(reference.state_dict()), {"logit_scale", *["model."+k for k in reference.model.state_dict()]})

    def test_wavelet_odd_shapes_and_sre_right_angle_equivariance(self):
        wavelet, _, _ = make_module("wtconv")
        x = torch.randn(2, 1024, 11, 13)
        self.assertEqual(wavelet(x).shape, x.shape)
        sre = component("sre_conv", "SRE_Conv2d")(8, 8, 3, padding=1).eval()
        x = torch.randn(2, 8, 12, 12)
        for k in range(4):
            self.assertTrue(torch.allclose(sre(torch.rot90(x, k, (2, 3))), torch.rot90(sre(x), k, (2, 3)), atol=1e-5))

    def test_superglobal_leaves_outside_top20_unchanged(self):
        x = torch.nn.functional.normalize(torch.randn(80, 32), dim=-1)
        queries = torch.nn.functional.normalize(torch.randn(2, 32), dim=-1)
        ranks = torch.argsort(x @ queries.T, dim=0, descending=True)
        augmentation = component("mda", "MDescAug")(M=20, K=9, beta=.15)
        rerank = component("superglobal", "RerankwMDA")(M=20, K=9, beta=.15)
        result = rerank(ranks, *augmentation(x, queries, ranks))
        self.assertTrue(torch.equal(result[20:], ranks[20:]))
        for i in range(2):
            self.assertEqual(sorted(result[:, i].tolist()), list(range(80)))

    def test_r2former_pair_head_can_train_without_cuda(self):
        from sample4geo.evaluate.rerank import make_r2former
        head = make_r2former()
        local = torch.randn(2, 16, 131)
        local[:, :, :3] = torch.rand(2, 16, 3)
        score, final = head(torch.randn(2, 1024), local, torch.randn(2, 1024), local.clone())
        self.assertEqual(score.shape, (2,))
        score.sum().backward()
        self.assertTrue(torch.isfinite(final).all())
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters()))
