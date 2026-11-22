import math

import timm
import torch
from torch import nn
from torch.nn import functional as F

from .modules import make_module

MODEL_NAME = "convnext_base.fb_in22k_ft_in1k_384"


class TimmModel(nn.Module):
    """沿用原项目的 model.* 权重名；两路共享 ConvNeXt-B 和 GAP。"""

    def __init__(self, pretrained=False, module="none", module_alpha=0.1, freeze_backbone=False, auxiliary="none"):
        super().__init__()
        self.model = timm.create_model(
            MODEL_NAME, pretrained=pretrained, num_classes=0, global_pool="avg"
        )
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))
        self.module_name, self.freeze_backbone = module, freeze_backbone
        self.auxiliary_name = auxiliary
        if auxiliary != "none":
            from .auxiliary import TrainingObjective
            with torch.random.fork_rng(devices=[]):
                self.training_head = TrainingObjective(auxiliary, self.model.num_features)
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

    def training_loss(self, satellite, drone, criterion, weight):
        # 同一对只提取一次特征图；此入口不会由独立评估调用。
        sat_map, drone_map = self.model.forward_features(satellite), self.model.forward_features(drone)
        sat, dro = self.encode_map(sat_map), self.encode_map(drone_map)
        main = criterion(sat, dro, self.logit_scale.exp())
        auxiliary = self.training_head(sat_map, drone_map, sat, dro, self.logit_scale.exp(), criterion)
        return main + weight * auxiliary, main, auxiliary

    def data_config(self):
        return timm.data.resolve_model_data_config(self.model)


def load_model_state(model, state, initialize_module=False):
    if not initialize_module:
        model.load_state_dict(state, strict=True)
        return []
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed = ("feature_module.", "parallel_projection.", "module_scale", "training_head.")
    if unexpected or any(not key.startswith(allowed) for key in missing):
        raise RuntimeError(f"权重结构不匹配：missing={missing}, unexpected={unexpected}")
    return missing


def module_alpha_metadata(model, override=None):
    if not hasattr(model, "module_scale"):
        return {}
    if override is not None:
        if not math.isfinite(override):
            raise ValueError("module-alpha必须是有限数")
        with torch.no_grad():
            model.module_scale.fill_(override)
    return {"module_alpha_effective": model.module_scale.detach().item(), "module_alpha_overridden": override is not None}
