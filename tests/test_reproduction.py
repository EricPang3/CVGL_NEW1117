"""训练行为契约：参考调度、随机状态及真实反向更新。"""

import random
import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sample4geo.dataset import UniqueBatchSampler, get_transforms
from sample4geo.loss import InfoNCE
from sample4geo.trainer import cosine_scheduler, train_epoch


class ReproductionTests(unittest.TestCase):
    def test_reference_cosine_landmarks(self):
        parameter = nn.Parameter(torch.zeros(()))
        optimizer = torch.optim.SGD([parameter], lr=1)
        scheduler = cosine_scheduler(optimizer, 10, {"epochs": 1, "warmup_epochs": 0.2})
        rates = [optimizer.param_groups[0]["lr"]]
        for _ in range(10):
            optimizer.step()
            scheduler.step()
            rates.append(optimizer.param_groups[0]["lr"])
        for step, expected in [(0, 0), (1, 0.5), (2, 1), (6, 0.5), (10, 0)]:
            self.assertAlmostEqual(rates[step], expected)

    def test_global_sampler_starts_at_configured_seed(self):
        pairs = [(str(i % 10), "sat", str(i)) for i in range(100)]
        random.seed(1)
        first = UniqueBatchSampler(pairs, 4)
        second = UniqueBatchSampler(pairs, 4)
        self.assertEqual(first.batches, UniqueBatchSampler(pairs, 4, 1).batches)
        self.assertNotEqual(first.batches, second.batches)

    def test_sues_augmentation_is_explicit(self):
        config = {"img_size": 384, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
        sat, drone = get_transforms({**config, "augmentation": "camp"}, training=True)
        original, _ = get_transforms({**config, "augmentation": "s4g"}, training=True)
        for transform, expected in ((sat, (0.7, 1.3)), (drone, (0.3, 1.7)), (original, (0.85, 1.15))):
            for actual, value in zip(transform.transforms[2].contrast, expected):
                self.assertAlmostEqual(actual, value)

    def test_train_updates_and_records_reference_first_lr(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Linear(8, 8)
                self.logit_scale = nn.Parameter(torch.tensor(1.0))

            def forward(self, a, b):
                return self.encoder(a), self.encoder(b)

        torch.manual_seed(1)
        model = Model()
        before = model.encoder.weight.detach().clone()
        pairs = [(str(i), "sat", str(i)) for i in range(8)]
        loader = DataLoader(TensorDataset(torch.randn(8, 8), torch.randn(8, 8), torch.arange(8)),
                            batch_sampler=UniqueBatchSampler(pairs, 4, 1))
        config = {"epochs": 1, "warmup_epochs": 0.1, "clip_grad": 100}
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        scheduler = cosine_scheduler(optimizer, 2, config)
        result = train_epoch(model, loader, optimizer, InfoNCE(), torch.cuda.amp.GradScaler(enabled=False),
                             scheduler, config, 1, torch.device("cpu"))
        self.assertEqual(result["first_lr"], 0)
        self.assertEqual(result["optimizer_updates"], 2)
        self.assertEqual(result["amp_skipped_steps"], 0)
        self.assertEqual(result["pairs"], 8)
        self.assertFalse(torch.equal(before, model.encoder.weight))
