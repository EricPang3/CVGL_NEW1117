"""将作者组件接到 ConvNeXt 末级；缩小配置和接口变化由实验配方明确记录。"""

import importlib

import torch
from torch import nn
from torch.nn import functional as F


def component(name, symbol):
    return getattr(importlib.import_module(f"third_party.components.{name}.layer"), symbol)


class Gate(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        return x * self.module(x)


class ChannelLast(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        return self.module(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class Float32Statistics(nn.Module):
    """保留原FP32统计语义；防止SelfNorm的1e-12方差eps在FP16下消失。"""
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            return self.module(x.float())


class ProjectedPAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.down = nn.Conv2d(channels, 128, 1)
        self.pam = component("danet_pam", "PAM_Module")(128)
        self.up = nn.Conv2d(128, channels, 1, bias=False)

    def forward(self, x):
        reduced = self.down(x)
        return x + self.up(self.pam(reduced) - reduced)


class PooledBranch(nn.Module):
    def __init__(self, name, channels):
        super().__init__()
        self.name = name
        self.dimensions = channels
        if name == "gpo":
            self.pool = component("gpo", "GPO")(32, 32)
        elif name == "salad":
            self.pool = component("salad", "SALAD")(num_channels=channels, num_clusters=8, cluster_dim=32, token_dim=128, dropout=0)
            self.dimensions = 384
        elif name == "gem":
            self.p = nn.Parameter(torch.tensor(3.0))
        elif name == "clip_attnpool":
            self.down = nn.Conv2d(channels, 256, 1)
            self.pool = component("clip_attnpool", "AttentionPool2d")(12, 256, 8, channels)
        elif name == "netvlad":
            self.down = nn.Conv2d(channels, 64, 1)
            self.pool = component("netvlad", "NetVLAD")(num_clusters=8, dim=64, use_faiss=False)
            self.dimensions = 512
        elif name == "isqrt_cov":
            self.pool = component("isqrt_cov", "MPNCOV")(iterNum=5, input_dim=channels, dimension_reduction=32)
            self.dimensions = 528

    def forward(self, x):
        if self.name == "gpo":
            tokens = x.flatten(2).transpose(1, 2).float()
            with torch.autocast(device_type=x.device.type, enabled=False):
                return self.pool(tokens, torch.full((len(x),), tokens.shape[1], device=x.device, dtype=torch.long))[0]
        if self.name == "salad":
            with torch.autocast(device_type=x.device.type, enabled=False):
                return self.pool((x.float(), x.float().mean((2, 3))))
        if self.name in ("gem", "gem_clipped_avg"):
            with torch.autocast(device_type=x.device.type, enabled=False):
                power = self.p.clamp(1, 8) if self.name == "gem" else 1
                return x.float().clamp_min(1e-6).pow(power).mean((2, 3)).pow(1 / power)
        if self.name == "rmac":
            return component("gem_rmac", "rmac")(x, L=3).flatten(1)
        if self.name == "clip_attnpool":
            return self.pool(F.adaptive_avg_pool2d(self.down(x), (12, 12)))
        if self.name == "netvlad":
            return self.pool(self.down(x))
        if self.name == "isqrt_cov":
            with torch.autocast(device_type=x.device.type, enabled=False):
                return self.pool(x.float()).flatten(1)
        raise ValueError(self.name)


class EDTHead(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.proj = nn.Linear(channels, 128)
        layer = component("saca", "SA_CA")(d_model=128, nhead=4, dropout=0, batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, num_layers=1)
        self.queries = nn.Parameter(torch.randn(1, 8, 128) * 1e-6)
        self.channel_proj = nn.Linear(128, 128)
        self.row_proj = nn.Linear(8, 2)

    def forward(self, x):
        tokens = x.flatten(2).transpose(1, 2)
        tokens = torch.cat((tokens.mean(1, keepdim=True), tokens), dim=1)
        out = self.decoder(self.queries.expand(len(x), -1, -1), self.proj(tokens))
        return self.row_proj(self.channel_proj(out).transpose(1, 2)).flatten(1)


def additional_module(name, c):
    enhancement = "enhancement"
    if name in ("fcanet", "fcanet_dc"):
        return component("fcanet", "MultiSpectralAttentionLayer")(c, 7, 7, reduction=16, freq_sel_method="top8" if name == "fcanet" else "top1"), enhancement, c
    if name == "coordatt":
        return component(name, "CoordAtt")(c, c), enhancement, c
    if name == "selfnorm":
        return Float32Statistics(component(name, "SelfNorm")(c)), enhancement, c
    if name == "se":
        from torchvision.ops import SqueezeExcitation
        return SqueezeExcitation(c, c // 16), enhancement, c
    if name == "cbam":
        return component(name, "CBAM")(c, reduction_ratio=16), enhancement, c
    if name == "triplet_attn":
        return component(name, "TripletAttention")(no_spatial=False), enhancement, c
    if name == "gcblock":
        return component(name, "ContextBlock")(c, ratio=1/16, pooling_type="att", fusion_types=("channel_add",)), enhancement, c
    if name == "grn":
        return ChannelLast(component(name, "GRN")(c)), enhancement, c
    if name == "danet_pam":
        return ProjectedPAM(c), enhancement, c
    if name == "caa":
        return Gate(component(name, "CAA")(c, h_kernel_size=7, v_kernel_size=7)), enhancement, c
    if name == "osblock":
        return component(name, "OSBlock")(c, c), enhancement, c
    if name == "inception_dw":
        return component(name, "InceptionDWConv2d")(c, band_kernel_size=7), enhancement, c
    if name == "uni_replk":
        return component(name, "DilatedReparamBlock")(c, kernel_size=7, deploy=False, attempt_use_lk_impl=False), "residual", c
    if name in ("acb", "repvgg_block", "sre_conv"):
        if name == "acb":
            inner = component(name, "ACBlock")(128, 128, 3, padding=1)
        elif name == "repvgg_block":
            inner = component(name, "RepVGGBlock")(128, 128, 3, padding=1, use_se=False)
        else:
            inner = component(name, "SRE_Conv2d")(128, 128, 3, padding=1)
        return nn.Sequential(nn.Conv2d(c, 128, 1), inner, nn.GELU(), nn.Conv2d(128, c, 1)), "residual", c
    if name == "ppm":
        return nn.Sequential(component(name, "PPM")(c, 128, bins=(1, 2, 3, 6)), nn.Conv2d(c + 512, c, 1), nn.GELU()), "residual", c
    if name == "odconv":
        return nn.Sequential(component(name, "ODConv2d")(c, c, 3, padding=1, groups=c, kernel_num=2), nn.GELU()), "residual", c
    if name == "edtformer_head":
        return EDTHead(c), "parallel", 256
    if name in ("gpo", "salad", "gem", "gem_clipped_avg", "clip_attnpool", "netvlad", "rmac", "isqrt_cov"):
        branch = PooledBranch(name, c)
        return branch, "parallel", branch.dimensions
    if name == "dwconv_budget":
        return nn.Sequential(*[nn.Conv2d(c, c, 3, padding=1, groups=c) for _ in range(5)], nn.GELU()), "residual", c
    raise ValueError(f"尚未实现的模块：{name}")
