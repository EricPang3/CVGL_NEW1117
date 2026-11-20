# Definitions extracted from https://raw.githubusercontent.com/sail-sg/inceptionnext/main/models/inceptionnext.py
# Upstream SHA256: c734f4efb2d3ae7041045bd5104548a7ccc555584b832ee4b3c1a72437ba0d3c
import math, copy, functools
from typing import Optional, Tuple, List, Union
import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.autograd import Function
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from timm.layers import to_2tuple, trunc_normal_
class InceptionDWConv2d(nn.Module):
    """ Inception depthweise convolution
    """
    def __init__(self, in_channels, square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125):
        super().__init__()
        
        gc = int(in_channels * branch_ratio) # channel numbers of a convolution branch
        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size//2, groups=gc)
        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size//2), groups=gc)
        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size//2, 0), groups=gc)
        self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)
        
    def forward(self, x):
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        return torch.cat(
            (x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)), 
            dim=1,
        )
