import torch.nn as nn
import torch.nn.functional as F
import math


class LIPooling(nn.Module):
    def __init__(self, child):
        super(LIPooling, self).__init__()
        self.opration = child
        self.sum = 0
        self.input = 0
        self.sumspike = 0

    def forward(self, x):
        batch, channel, w, h = x.shape
        k, s, p, cm = self.opration.kernel_size, self.opration.stride, self.opration.padding, self.opration.ceil_mode

        if cm is True:
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
