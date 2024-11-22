import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy


class Quantization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, constant=100):
        ctx.constant = constant
        return torch.div(torch.floor(torch.mul(tensor, constant)), constant)

    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output), None


Quantization_ = Quantization.apply


class ClipQuan(nn.Module):
    def __init__(self, cq_level: int = 100):
        super(ClipQuan, self).__init__()
        self.min = 0.
        self.max = 1.
        self.cq_level = cq_level
        self.in_record = 0.
        self.out_record = 0.

    def forward(self, x):
        # self.in_record = deepcopy(x.detach())

        x = torch.clamp(x, min=self.min, max=self.max)
        x = Quantization_(x, self.cq_level)

        # self.out_record = deepcopy(x.detach())
        return x


def replace_relu_by_cqrelu(model, cq_level=None):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = replace_relu_by_cqrelu(module, cq_level)
        if 'relu' in module.__class__.__name__.lower():
                model._modules[name] = ClipQuan(cq_level=cq_level)
    return model