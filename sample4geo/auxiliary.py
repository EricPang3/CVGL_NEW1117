"""仅训练期使用成对特征；评估时每张图独立编码，不提供真值配对。"""

import torch
from torch import nn
from torch.nn import functional as F

from .research_modules import SemanticParts

AUXILIARIES = ("none", "consensus", "visible_consensus", "part_align", "view_relation", "vicreg")


class TrainingObjective(nn.Module):
    def __init__(self, name, channels):
        super().__init__()
        if name not in AUXILIARIES[1:]:
            raise ValueError(f"未知辅助目标：{name}")
        self.name = name
        if name in ("consensus", "visible_consensus", "part_align"):
            self.parts = SemanticParts(channels)
        if name in ("consensus", "visible_consensus"):
            self.project = nn.Sequential(nn.Linear(128, channels), nn.LayerNorm(channels))

    def forward(self, satellite_map, drone_map, sat, drone, scale, criterion):
        with torch.autocast(device_type=sat.device.type, enabled=False):
            sat, drone, scale = sat.float(), drone.float(), scale.float()
            if self.name in ("consensus", "visible_consensus", "part_align"):
                a, am, _ = self.parts(satellite_map)
                b, bm, _ = self.parts(drone_map)
                if self.name == "part_align":
                    return torch.stack([criterion(a[:, k], b[:, k], scale) for k in range(a.shape[1])]).mean()
                weights = torch.ones_like(am)
                if self.name == "visible_consensus":
                    # 语义相似/部件占比代理，不声称具有真实深度几何可见性。
                    agreement = (F.normalize(a, dim=-1) * F.normalize(b, dim=-1)).sum(-1)
                    weights = (0.1 + agreement.sigmoid()) * (am * bm).clamp_min(1e-6).sqrt()
                weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-6)
                joint = self.project(((a + b) * 0.5 * weights.unsqueeze(-1)).sum(1))
                return (criterion(sat, joint, scale) + criterion(drone, joint, scale)) / 2
            if self.name == "view_relation":
                # 每个batch地点唯一；比较两个视角的地点关系，不将额外同地点图片当负例。
                a, b = F.normalize(sat, dim=-1), F.normalize(drone, dim=-1)
                mask = torch.eye(len(a), device=a.device, dtype=torch.bool)
                p = ((a @ a.T / 0.2).masked_fill(mask, -1e4)).softmax(-1).clamp_min(1e-8)
                q = ((b @ b.T / 0.2).masked_fill(mask, -1e4)).softmax(-1).clamp_min(1e-8)
                mean = (p + q) / 2
                return (p * (p.log() - mean.log()) + q * (q.log() - mean.log())).sum(-1).mean() / 2
            # VICReg风格统计约束，使用未归一化描述子；不是新的论文复现。
            penalties = []
            for values in (sat, drone):
                centered = values - values.mean(0)
                variance = F.relu(1 - (centered.square().mean(0) + 1e-4).sqrt()).mean()
                covariance = centered.T @ centered / max(len(values) - 1, 1)
                covariance = covariance - torch.diag_embed(covariance.diag())
                penalties.append(variance + covariance.square().sum() / values.shape[1])
            return torch.stack(penalties).mean()
