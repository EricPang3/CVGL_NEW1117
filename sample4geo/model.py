import math

import timm
import torch
from torch import nn

MODEL_NAME = "convnext_base.fb_in22k_ft_in1k_384"


class TimmModel(nn.Module):
    """沿用原项目的 model.* 权重名；两路共享 ConvNeXt-B 和 GAP。"""

    def __init__(self, pretrained=False):
        super().__init__()
        self.model = timm.create_model(
            MODEL_NAME, pretrained=pretrained, num_classes=0, global_pool="avg"
        )
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def forward(self, image, other=None):
        features = self.model(image)
        return features if other is None else (features, self.model(other))

    def data_config(self):
        return timm.data.resolve_model_data_config(self.model)
