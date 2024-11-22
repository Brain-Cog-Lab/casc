import math
import numpy as np
import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from copy import deepcopy


class BaseNeuron(nn.Module):
    def __init__(self,
                 act_func=None,
                 threshold: float = 1.,
                 soft_mode: bool = False,
                 **kwargs
                 ):
        super(BaseNeuron, self).__init__()
        self.threshold = Parameter(torch.tensor(threshold), requires_grad=False)
        self.mem = 0.
        self.spike = 0.
        self.summem = 0.
        self.sumspike = 0.
        self.record_sum = False
        self.v_reset = 0.
        self.soft_mode = soft_mode
        self.act_func = act_func
        self.t = 0

    def cal_spike(self):
        if self.act_func is not None:
            self.spike = self.act_func(self.mem - self.threshold)
        else:
            self.spike = (self.mem - self.threshold >= 0.).to(self.mem.dtype)

    def cal_mem(self, x):
        raise NotImplementedError

    def hard_reset(self):
        self.mem = self.mem * (1 - self.spike)

    def soft_reset(self):
        self.mem = self.mem - self.threshold * self.spike.detach()

    def forward(self, x):
        self.cal_mem(x)
        self.cal_spike()
        if self.record_sum:
            self.summem += x.detach()
            self.sumspike += self.spike.detach()
        self.soft_reset() if self.soft_mode else self.hard_reset()

        return self.spike

    def reset(self):
        self.mem = self.v_reset
        self.spike = 0.

        self.summem = 0.
        self.sumspike = 0.
        self.t = 0

    def set_threshold(self, threshold):
        self.threshold = Parameter(torch.tensor(threshold, dtype=torch.float), requires_grad=False)

    def set_tau(self, tau):
        if hasattr(self, 'tau'):
            self.tau = Parameter(torch.tensor(tau, dtype=torch.float), requires_grad=False)
        else:
            raise NotImplementedError


class IF(BaseNeuron):
    def __init__(self, act_func=None, threshold=1., **kwargs):
        super().__init__(act_func, threshold, **kwargs)

    def cal_mem(self, x):
        self.mem = self.mem + x


class LIF(BaseNeuron):
    def __init__(self, act_func, threshold=1., **kwargs):
        super().__init__(act_func, threshold, **kwargs)
        self.tau = kwargs['tau']

    def cal_mem(self, x):
        # self.mem = self.mem * (1 - 1. / self.tau) + x
        self.mem = self.mem + (x - self.mem) / self.tau


class Burst(BaseNeuron):
    def __init__(self, act_func=None, threshold=1., **kwargs):
        super().__init__(act_func, threshold, **kwargs)
        self.gamma = kwargs['gamma'] if 'gamma' in kwargs else 1
        self.soft_mode = kwargs['soft_mode'] if 'soft_mode' in kwargs else False
        self.neg = kwargs['neg'] if 'neg' in kwargs else False
        self.sleep = False

    def cal_mem(self, x):
        self.mem = self.mem + x
        self.summem += x

    def cal_spike(self):
        if not self.sleep:
            self.t += 1
        if not self.neg:
            self.spike = torch.div(self.mem, self.threshold, rounding_mode='floor').clamp(min=0, max=self.gamma)
        else:
            self.spike = torch.div(self.mem, self.threshold, rounding_mode='floor').clamp(min=-self.gamma, max=self.gamma)
            self.spike[(self.spike <= 0) & (self.sumspike <= 0)] = 0
        self.spike[(self.spike > 0) & (self.sumspike >= self.t)] = 0

        self.sumspike += self.spike

