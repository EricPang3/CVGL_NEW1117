"""2026研究候选：论文思想的轻量适配，不是作者完整实现。来源见configs/research_sources.yaml。"""

import torch
from torch import nn
from torch.nn import functional as F


FEATURE_MODULES = ("prototype_pool", "prototype_graph", "prototype_dustbin", "scale_gate",
                   "scale_uniform", "frequency_gate", "spatial_smooth", "wt_proto_graph", "dw_proto_graph")


class SemanticParts(nn.Module):
    """共享原型竞争分配；不要求两个视角的像素坐标对齐。"""
    def __init__(self, channels, dimension=128, parts=8, dustbin=False):
        super().__init__()
        self.parts, self.dustbin = parts, dustbin
        self.project = nn.Conv2d(channels, dimension, 1)
        self.prototypes = nn.Parameter(torch.randn(parts + int(dustbin), dimension) * 0.02)

    def forward(self, x):
        # 投影、竞争softmax和除法统一FP32，避免训练后激活放大引发FP16溢出。
        with torch.autocast(device_type=x.device.type, enabled=False):
            tokens = self.project(x.float()).flatten(2).transpose(1, 2)
            scores = F.normalize(tokens, dim=-1) @ F.normalize(self.prototypes, dim=-1).T / 0.1
            assignment = scores.softmax(-1)[..., :self.parts].transpose(1, 2)
            mass = assignment.sum(-1, keepdim=True)
            weights = assignment / mass.clamp_min(1e-6)
            parts = weights @ tokens
            h, w = x.shape[-2:]
            y, z = torch.meshgrid(torch.linspace(-1, 1, h, device=x.device),
                                  torch.linspace(-1, 1, w, device=x.device), indexing="ij")
            centers = weights @ torch.stack((y, z), -1).reshape(-1, 2)
            return parts, mass.squeeze(-1) / tokens.shape[1], centers


class PrototypePool(nn.Module):
    def __init__(self, channels, graph=False, dustbin=False, prefilter=None):
        super().__init__()
        self.parts = SemanticParts(channels, dustbin=dustbin)
        self.prefilter = prefilter
        self.graph = graph
        if graph:
            self.position = nn.Linear(2, 128, bias=False)
            self.attention = nn.MultiheadAttention(128, 4, dropout=0, batch_first=True)
            self.norm = nn.LayerNorm(128)
        self.output = nn.Linear(8 * 128, 256, bias=False)

    def forward(self, x):
        if self.prefilter is not None:
            # 只在并行头内组合；原始GAP旁路始终保留。
            x = x + 0.1 * self.prefilter(x)
        with torch.autocast(device_type=x.device.type, enabled=False):
            parts, mass, centers = self.parts(x)
            if self.graph:
                positioned = parts + self.position(centers)
                parts = self.norm(parts + self.attention(positioned, positioned, parts, need_weights=False)[0])
            # dustbin丢弃的质量必须影响描述子，不能在归一化后被完全抵消。
            weighted = F.normalize(parts, dim=-1) * (mass * 8).clamp_min(1e-6).sqrt().unsqueeze(-1)
            return self.output(weighted.flatten(1))


class ScaleGate(nn.Module):
    """末级图的三种感受野；不是DINO-GFSA的跨层融合/Mamba实现。"""
    def __init__(self, channels, adaptive=True):
        super().__init__()
        self.down = nn.Conv2d(channels, 128, 1)
        self.branches = nn.ModuleList(nn.Conv2d(128, 128, 3, padding=d, dilation=d, groups=128) for d in (1, 2, 4))
        self.gate = nn.Conv2d(128, 3, 1) if adaptive else None
        self.up = nn.Conv2d(128, channels, 1, bias=False)

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            reduced = F.gelu(self.down(x.float()))
            branches = torch.stack([F.gelu(branch(reduced)) for branch in self.branches], 1)
            weights = self.gate(reduced).softmax(1).unsqueeze(2) if self.gate is not None else 1 / 3
            return self.up((branches * weights).sum(1))


class FrequencyGate(nn.Module):
    """可学习径向低/高频门控；非线性后再GAP，避免高频在线性GAP中恒为零。"""
    def __init__(self, channels):
        super().__init__()
        self.cutoff = nn.Parameter(torch.tensor(0.0))
        self.gains = nn.Parameter(torch.tensor([0.1, -0.1]).reshape(2, 1, 1, 1).expand(2, channels, 1, 1).clone())

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            value = x.float()
            h, w = value.shape[-2:]
            fy = torch.fft.fftfreq(h, device=x.device).reshape(-1, 1)
            fx = torch.fft.rfftfreq(w, device=x.device).reshape(1, -1)
            radius = (fx.square() + fy.square()).sqrt()
            low_mask = torch.sigmoid((self.cutoff.sigmoid() * 0.5 - radius) / 0.05)
            spectrum = torch.fft.rfft2(value.contiguous(), norm="ortho")
            low = torch.fft.irfft2(spectrum * low_mask, s=(h, w), norm="ortho")
            filtered = value + self.gains[0].tanh() * low + self.gains[1].tanh() * (value - low)
            return F.gelu(filtered) - F.gelu(value)


class SpatialSmooth(nn.Module):
    def __init__(self, channels):
        super().__init__()
        kernel = torch.tensor([1., 4., 6., 4., 1.])
        self.register_buffer("kernel", (kernel[:, None] * kernel[None, :] / 256).expand(channels, 1, 5, 5).clone())

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            value = x.float()
            blurred = F.conv2d(F.pad(value, (2, 2, 2, 2), mode="replicate"), self.kernel, groups=value.shape[1])
            return F.gelu(blurred) - F.gelu(value)


def research_module(name, channels):
    if name.startswith("prototype_"):
        return PrototypePool(channels, graph=name == "prototype_graph", dustbin=name == "prototype_dustbin"), "parallel", 256
    if name in ("wt_proto_graph", "dw_proto_graph"):
        from .modules import make_module
        prefilter, _, _ = make_module("wtconv" if name.startswith("wt_") else "dwconv", channels)
        return PrototypePool(channels, graph=True, prefilter=prefilter), "parallel", 256
    if name in ("scale_gate", "scale_uniform"):
        return ScaleGate(channels, adaptive=name == "scale_gate"), "residual", channels
    if name == "frequency_gate":
        return FrequencyGate(channels), "residual", channels
    if name == "spatial_smooth":
        return SpatialSmooth(channels), "residual", channels
    raise ValueError(f"未知研究模块：{name}")
