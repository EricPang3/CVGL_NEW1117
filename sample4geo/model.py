import math

import timm
import torch
from torch import nn
from torch.nn import functional as F

from .modules import make_module

MODEL_NAME = "convnext_base.fb_in22k_ft_in1k_384"


class TimmModel(nn.Module):
    """沿用原项目的 model.* 权重名；两路共享 ConvNeXt-B 和 GAP。"""

    def __init__(self, pretrained=False, module="none", module_alpha=0.1, freeze_backbone=False):
        super().__init__()
        self.model = timm.create_model(
            MODEL_NAME, pretrained=pretrained, num_classes=0, global_pool="avg"
        )
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))
        self.module_name, self.freeze_backbone = module, freeze_backbone
        if module != "none":
            # 新模块初始化不消耗后续数据加载使用的全局 RNG，方便配对 seed 对照。
            with torch.random.fork_rng(devices=[]):
                self.feature_module, self.module_mode, dimensions = make_module(module, self.model.num_features)
                if self.module_mode == "parallel":
                    self.parallel_projection = nn.Identity() if dimensions == self.model.num_features else nn.Linear(dimensions, self.model.num_features, bias=False)
            self.module_scale = nn.Parameter(torch.tensor(float(module_alpha)), requires_grad=module != "burst_gap")
        if freeze_backbone:
            self.model.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.model.eval()
        return self

    def encode_map(self, features):
        if self.module_name == "none":
            return self.model.forward_head(features)
        branch = self.feature_module(features)
        if self.module_mode == "parallel":
            branch = branch[0] if isinstance(branch, tuple) else branch
            original = self.model.forward_head(features)
            projected = self.parallel_projection(branch)
            residual = F.normalize(projected, dim=-1).to(original.dtype) * original.norm(dim=-1, keepdim=True)
            return original + self.module_scale.to(original.dtype) * residual
        branch = branch.to(features.dtype)
        if self.module_mode == "enhancement":
            adapted = features + self.module_scale * (branch - features)
        elif self.module_mode == "residual":
            adapted = features + self.module_scale * branch
        else:
            raise ValueError(self.module_mode)
        return self.model.forward_head(adapted)

    def encode(self, image):
        if self.module_name == "none":
            return self.model(image)
        return self.encode_map(self.model.forward_features(image))

    def forward(self, image, other=None):
        features = self.encode(image)
        return features if other is None else (features, self.encode(other))

    def data_config(self):
        return timm.data.resolve_model_data_config(self.model)


def load_model_state(model, state, initialize_module=False):
    if not initialize_module:
        model.load_state_dict(state, strict=True)
        return []
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed = ("feature_module.", "parallel_projection.", "module_scale")
    if unexpected or any(not key.startswith(allowed) for key in missing):
        raise RuntimeError(f"权重结构不匹配：missing={missing}, unexpected={unexpected}")
    return missing
