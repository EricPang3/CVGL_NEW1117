import math
import torch
import torch.nn.functional as F




def create_2d_wavelet_filter(wave, in_size, out_size, type=torch.float):
    # CVGL adaptation: closed-form db1/Haar only, exactly the default upstream filters.
    if wave != 'db1': raise ValueError('CVGL WTConv only supports the registered db1 configuration')
    lo=torch.tensor([1/math.sqrt(2),1/math.sqrt(2)],dtype=type)
    hi=torch.tensor([1/math.sqrt(2),-1/math.sqrt(2)],dtype=type)
    filters=torch.stack([lo[:,None]*lo[None,:],lo[:,None]*hi[None,:],hi[:,None]*lo[None,:],hi[:,None]*hi[None,:]])[:,None]
    return filters.repeat(in_size,1,1,1), filters.repeat(out_size,1,1,1)






def wavelet_2d_transform(x, filters):
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x


def inverse_2d_wavelet_transform(x, filters):
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    return x
