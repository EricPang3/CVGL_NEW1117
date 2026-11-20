# Definitions extracted from https://raw.githubusercontent.com/facebookresearch/ConvNeXt-V2/main/models/utils.py
# Upstream SHA256: f7eca4be0696ccdae8b7a05387870cacc66d6408a7acc44b0b7f1ae56b2df631
import math, copy, functools
from typing import Optional, Tuple, List, Union
import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.autograd import Function
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from timm.layers import to_2tuple, trunc_normal_
class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x
