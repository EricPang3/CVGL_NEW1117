import math, copy, functools
from typing import Optional, Tuple, List, Union
import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.autograd import Function
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from timm.layers import to_2tuple, trunc_normal_

class BaseModule(nn.Module):
    def __init__(self, init_cfg=None): super().__init__()
def constant_init(module, val, bias=0):
    nn.init.constant_(module.weight, val)
    if getattr(module, 'bias', None) is not None: nn.init.constant_(module.bias, bias)
def kaiming_init(module, mode='fan_out'):
    nn.init.kaiming_normal_(module.weight, mode=mode)
    if getattr(module, 'bias', None) is not None: nn.init.zeros_(module.bias)
def ConvModule(inp,out,kernel,stride=1,padding=0,groups=1,norm_cfg=None,act_cfg=None):
    layers=[nn.Conv2d(inp,out,kernel,stride,padding,groups=groups,bias=norm_cfg is None)]
    if norm_cfg: layers.append(nn.BatchNorm2d(out,eps=norm_cfg.get('eps',1e-5),momentum=norm_cfg.get('momentum',.1)))
    if act_cfg: layers.append(nn.SiLU(inplace=True))
    return nn.Sequential(*layers)
