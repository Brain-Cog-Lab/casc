import math
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F


class LIPool(nn.Module):
    r"""Precise replacement of the largest pooling layer for the transformation method
    LIPooling ensures that the maximum output value in the converted SNN is the same as the expected value
    by introducing a lateral suppression mechanism.

    Reference:
    https://arxiv.org/abs/2204.13271
    """

    def __init__(self, child):
        super(LIPool, self).__init__()
        if child is None:
            raise NotImplementedError("child should be Pooling operation with torch")

        self.opration = child
        self.sumspike = 0

    def forward(self, x):
        batch, channel, w, h = x.shape
        k, s, p, cm = self.opration.kernel_size, self.opration.stride, self.opration.padding, self.opration.ceil_mode
        if cm:
            out_w = math.ceil((w + 2 * p - (k - 1) - 1) / s) + 1
            out_h = math.ceil((h + 2 * p - (k - 1) - 1) / s) + 1
            pad_w = round((out_w - 1 - (w + 2 * p - (k - 1) - 1) / s) * s)
            pad_h = round((out_h - 1 - (h + 2 * p - (k - 1) - 1) / s) * s)
            x = F.pad(x, (0, pad_w, 0, pad_h))
        else:
            out_w = math.floor((w + 2 * p - (k - 1) - 1) / s) + 1
            out_h = math.floor((h + 2 * p - (k - 1) - 1) / s) + 1
        a = F.unfold(x, kernel_size=k, stride=s, padding=p)
        self.sumspike += a.view(batch, channel, k * k, out_h, out_w)
        out = self.sumspike.max(dim=2, keepdim=True)[0]
        self.sumspike -= out
        out = out.squeeze(2)

        return out

    def reset(self):
        self.sumspike = 0


class SMaxPool(nn.Module):
    r"""The normal replacement of the maximum pooling layer for the transformation method
    Selecting neurons with the maximum pulse rate can satisfy the need of general maximum pooling layer

    Reference:
    https://arxiv.org/abs/1612.04052
    """

    def __init__(self, child):
        super(SMaxPool, self).__init__()
        self.opration = child
        self.sumspike = 0

    def forward(self, x):
        self.sumspike += x
        single = self.opration(self.sumspike * 1000)
        sum_plus_spike = self.opration(x + self.sumspike * 1000)

        return sum_plus_spike - single

    def reset(self):
        self.sumspike = 0