"""独立的共享末级模块；默认 none 不改变原编码器。来源见 third_party/SOURCES.json。"""

import torch
from torch import nn
from torch.nn import functional as F


class BurstGAP(nn.Module):
    """借用重复区域降权思想，不声称实现完整 VLAD-BuFF；固定规则，不拟合测试集。"""
    def weights(self, x):
        tokens = F.normalize(x.flatten(2).float(), dim=1)
        similarity = tokens.transpose(1, 2) @ tokens
        n = similarity.shape[-1]
        mask = ~torch.eye(n, device=x.device, dtype=torch.bool)
        count = (torch.sigmoid((similarity - 0.7) / 0.1) * mask).sum(-1)
        self.last_density = (count.mean(-1) / max(n - 1, 1)).detach()
        weights = (1 + count).rsqrt()
        return weights / weights.mean(-1, keepdim=True)

    def forward(self, x):
        return x * self.weights(x).reshape(x.shape[0], 1, *x.shape[-2:]).to(x.dtype)


class DepthwiseResidual(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)

    def forward(self, x):
        return F.gelu(self.conv(x))


def make_module(name, channels=1024):
    from .research_modules import FEATURE_MODULES, research_module
    if name in FEATURE_MODULES:
        return research_module(name, channels)
    if name == "burst_gap":
        return BurstGAP(), "enhancement", channels
    if name == "eca":
        from third_party.ECANet.eca_module import eca_layer
        return eca_layer(channels, k_size=5), "enhancement", channels
    if name == "dwconv":
        return DepthwiseResidual(channels), "residual", channels
    if name == "wtconv":
        from third_party.WTConv.wtconv.wtconv2d import WTConv2d
        return nn.Sequential(WTConv2d(channels, channels, kernel_size=3, stride=1, wt_levels=1), nn.GELU()), "residual", channels
    if name == "boq":
        from third_party.Bag_of_Queries.boq import BoQ
        return BoQ(channels, proj_channels=128, num_queries=8, num_layers=1, row_dim=2), "parallel", 256
    from .module_adapters import additional_module
    return additional_module(name, channels)
