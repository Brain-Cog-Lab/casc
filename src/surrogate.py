import math

import torch
from torch import nn
from torch.nn import functional as F


def heaviside(x):
    return (x >= 0.).to(x.dtype)


class SurrogateFunctionBase(nn.Module):
    def __init__(self, alpha=2., requires_grad=True):
        super().__init__()
        self.alpha = nn.Parameter(
            torch.tensor(alpha, dtype=torch.float),
            requires_grad=requires_grad)


    @staticmethod
    def act_fun(x, alpha):
        raise NotImplementedError

    def forward(self, x):
        return self.act_fun(x, self.alpha)


class sigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        if ctx.needs_input_grad[0]:
            s_x = torch.sigmoid(ctx.alpha * ctx.saved_tensors[0])
            grad_x = grad_output * s_x * (1 - s_x) * ctx.alpha
        return grad_x, None


class SigmoidGrad(SurrogateFunctionBase):
    def __init__(self, alpha=1., requires_grad=False, **kwargs):
        super().__init__(alpha, requires_grad)

    @staticmethod
    def act_fun(x, alpha):
        return sigmoid.apply(x, alpha)


class delta(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        if x.requires_grad:
            ctx.save_for_backward(x)

        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = grad_output * (ctx.saved_tensors[0] == 0.).float()
        return grad_x


class DeltaGrad(SurrogateFunctionBase):
    def __init__(self, **kwargs):
        super().__init__()

    @staticmethod
    def act_fun(x):
        return delta.apply(x)

    def forward(self, x):
        return self.act_fun(x)



class step(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        if x.requires_grad:
            ctx.save_for_backward(x)
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = grad_output * (ctx.saved_tensors[0] >= 0.).float()
        return grad_x


class StepGrad(SurrogateFunctionBase):
    def __init__(self, **kwargs):
        super().__init__()

    @staticmethod
    def act_fun(x):
        return step.apply(x)

    def forward(self, x):
        return self.act_fun(x)


class realsigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return torch.sigmoid(alpha * x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        if ctx.needs_input_grad[0]:
            s_x = torch.sigmoid(ctx.alpha * ctx.saved_tensors[0])
            grad_x = grad_output * s_x * (1 - s_x) * ctx.alpha
        return grad_x, None


class RealSigmoidGrad(SurrogateFunctionBase):
    def __init__(self, alpha=1., requires_grad=False, **kwargs):
        super().__init__(alpha, requires_grad)

    @staticmethod
    def act_fun(x, alpha):
        return realsigmoid.apply(x, alpha)


class rectangular(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = grad_output * ctx.alpha * heaviside(1. / ctx.alpha - ctx.saved_tensors[0].abs())
        return grad_x, None


class RectangularGrad(SurrogateFunctionBase):
    def __init__(self, alpha=1., requires_grad=False, **kwargs):
        super().__init__(alpha, requires_grad)

    @staticmethod
    def act_fun(x, alpha):
        return rectangular.apply(x, alpha)


class piecewiselinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        if ctx.needs_input_grad[0]:
            mask_zero = (ctx.saved_tensors[0].abs() >= 1./ctx.alpha)
            grad_x = -ctx.alpha * ctx.alpha * ctx.saved_tensors[0].abs() + ctx.alpha
            grad_x.masked_fill_(mask_zero, 0)
            grad_x = grad_output * grad_x
        return grad_x, None


class PiecewiseLinearGrad(SurrogateFunctionBase):
    def __init__(self, alpha=1., requires_grad=False, **kwargs):
        super().__init__(alpha, requires_grad)

    @staticmethod
    def act_fun(x, alpha):
        return piecewiselinear.apply(x, alpha)


class probpiecewiselinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        if ctx.needs_input_grad[0]:
            mask0 = (ctx.saved_tensors[0] >= 0).float()
            mask_zero = (ctx.saved_tensors[0].abs() >= 1./ctx.beta)
            grad_x = -ctx.alpha * ctx.beta * ctx.saved_tensors[0].abs() + ctx.alpha
            grad_x.masked_fill_(mask_zero, 0)

            rand_mask = (torch.rand_like(mask0) <= 0.2).float().to(grad_x.device)
            grad_x = grad_x * (1-rand_mask) #+ mask0 * rand_mask

            grad_x = grad_output * grad_x
        return grad_x, None


class ProbPiecewiseLinearGrad(SurrogateFunctionBase):
    def __init__(self, alpha=1., requires_grad=False, **kwargs):
        super().__init__(alpha, requires_grad)

    @staticmethod
    def act_fun(x, alpha):
        return probpiecewiselinear.apply(x, alpha)


class piecewiseexp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = ctx.alpha * torch.exp(-ctx.alpha * ctx.saved_tensors[0].abs())
            grad_x = grad_output * grad_x
        return grad_x, None


class PiecewiseExpGrad(SurrogateFunctionBase):
    def __init__(self, alpha=1., requires_grad=False, **kwargs):
        super().__init__(alpha, requires_grad)

    @staticmethod
    def act_fun(x, alpha):
        return piecewiseexp.apply(x, alpha)


class atan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        shared_c = grad_output / (1 + (ctx.alpha * math.pi /2 * ctx.saved_tensors[0]).square())
        if ctx.needs_input_grad[0]:
            grad_x = ctx.alpha * shared_c

        return grad_x, None


class AtanGrad(SurrogateFunctionBase):
    def __init__(self, alpha=2., requires_grad=True, **kwargs):
        super().__init__(alpha, requires_grad)

    @staticmethod
    def act_fun(x, alpha):
        return atan.apply(x, alpha)


def get_mask(mix_mode, sub_prob, x=None):
    if mix_mode == 'rand':
        rand_mask = (torch.rand_like(x) <= sub_prob).float().to(x.device)
        return rand_mask

def get_sub_func_grad(x, sub_func, alpha=2.):
    if sub_func == 'StepGrad':
        sub_func_grad = (x >= 0).float()
    elif sub_func == 'DeltaGrad':
        sub_func_grad = (x == 0).float()
    elif sub_func == 'ZeroGrad':
        sub_func_grad = torch.zeros_like(x).to(x.device)
    elif sub_func == 'RectangularGrad':
        sub_func_grad = alpha * heaviside(1. / alpha - x.abs())
    return sub_func_grad


class mixedplgrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha, sub_func, sub_prob, mix_mode):
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        ctx.sub_func = sub_func
        ctx.sub_prob = sub_prob
        ctx.mix_mode = mix_mode
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        if ctx.needs_input_grad[0]:
            mask_zero = (ctx.saved_tensors[0].abs() >= 1. / ctx.alpha)
            grad_x = -ctx.alpha * ctx.alpha * ctx.saved_tensors[0].abs() + ctx.alpha
            grad_x.masked_fill_(mask_zero, 0)

            sub_func_grad = get_sub_func_grad(ctx.saved_tensors[0], ctx.sub_func, ctx.alpha)
            mask = get_mask(ctx.mix_mode, ctx.sub_prob, ctx.saved_tensors[0])
            grad_x = grad_x * (1 - mask)  + mask * sub_func_grad
            grad_x = grad_output * grad_x
        return grad_x, None, None, None, None


class MixedPLGrad(SurrogateFunctionBase):
    def __init__(self, alpha=2., requires_grad=False, **kwargs):
        super().__init__(alpha, requires_grad)
        self.sub_func = kwargs['sub_func'] if "sub_func" in kwargs else 'step'
        self.sub_prob = kwargs['sub_prob'] if "sub_prob" in kwargs else 0.2
        self.mix_mode = kwargs['mix_mode'] if "mix_mode" in kwargs else 'rand'

    @staticmethod
    def act_fun(x, alpha, sub_func, sub_prob, mix_mode):
        return mixedplgrad.apply(x, alpha, sub_func, sub_prob, mix_mode)

    def forward(self, x):
        return self.act_fun(x, self.alpha, self.sub_func, self.sub_prob, self.mix_mode)


class mixedatangrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha, sub_func, sub_prob, mix_mode):
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        ctx.sub_func = sub_func
        ctx.sub_prob = sub_prob
        ctx.mix_mode = mix_mode
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        shared_c = grad_output / (1 + (ctx.alpha * math.pi / 2 * ctx.saved_tensors[0]).square())
        if ctx.needs_input_grad[0]:
            grad_x = ctx.alpha * shared_c

            sub_func_grad = get_sub_func_grad(ctx.saved_tensors[0], ctx.sub_func, ctx.alpha)
            mask = get_mask(ctx.mix_mode, ctx.sub_prob, ctx.saved_tensors[0])
            grad_x = grad_x * (1 - mask)  + mask * sub_func_grad
            grad_x = grad_output * grad_x
        return grad_x, None, None, None, None


class MixedAtanGrad(SurrogateFunctionBase):
    def __init__(self, alpha=2., requires_grad=False, **kwargs):
        super().__init__(alpha, requires_grad)
        self.sub_func = kwargs['sub_func'] if "sub_func" in kwargs else 'step'
        self.sub_prob = kwargs['sub_prob'] if "sub_prob" in kwargs else 0.2
        self.mix_mode = kwargs['mix_mode'] if "mix_mode" in kwargs else 'rand'

    @staticmethod
    def act_fun(x, alpha, sub_func, sub_prob, mix_mode):
        return mixedatangrad.apply(x, alpha, sub_func, sub_prob, mix_mode)

    def forward(self, x):
        return self.act_fun(x, self.alpha, self.sub_func, self.sub_prob, self.mix_mode)
