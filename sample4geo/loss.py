import torch
from torch import nn
from torch.nn import functional as F


class InfoNCE(nn.Module):
    """S4G 的 batch 内双向 InfoNCE；每个 batch 的地点必须互不重复。"""

    def __init__(self, label_smoothing=0.1):
        super().__init__()
        self.label_smoothing = label_smoothing

    def forward(self, satellite, drone, scale):
        logits = scale * F.normalize(satellite, dim=-1) @ F.normalize(drone, dim=-1).T
        target = torch.arange(len(logits), device=logits.device)
        return (
            F.cross_entropy(logits, target, label_smoothing=self.label_smoothing)
            + F.cross_entropy(logits.T, target, label_smoothing=self.label_smoothing)
        ) / 2
