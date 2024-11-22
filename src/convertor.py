import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .clipquantization import ClipQuan
from .utils import mergeConvBN
from .pool import LIPool, SMaxPool
from .neuron import Burst
import types
from copy import deepcopy


class HookScale(nn.Module):
    """ 在每个ReLU层后记录该层的百分位最大值

    For channelnorm: 获取最大值时使用了torch.quantile
    For layernorm：  使用sort，然后手动取百分比，因为quantile在计算单个通道时有上限，batch较大时易出错
    """
    def __init__(self,
                 p: float = 0.9995,
                 channelnorm: bool = False,
                 gamma: float = 0.999,
                 ):
        super().__init__()
        if channelnorm:
            self.register_buffer('scale', torch.tensor(0.0))
        else:
            self.register_buffer('scale', torch.tensor(0.0))

        self.p = p
        self.channelnorm = channelnorm
        self.gamma = gamma

    def forward(self, x):
        x = torch.where(x.detach() < self.gamma, x.detach(), torch.tensor(self.gamma, dtype=x.dtype, device=x.device))
        if len(x.shape) == 4 and self.channelnorm:
            num_channel = x.shape[1]
            tmp = torch.quantile(x.permute(1, 0, 2, 3).reshape(num_channel, -1), self.p, dim=1,
                                        interpolation='lower') + 1e-10
            self.scale = torch.max(tmp, self.scale)
        else:
            sort, _ = torch.sort(x.view(-1))
            self.scale = torch.max(sort[int(sort.shape[0] * self.p) - 1], self.scale)
        return x


class Hookoutput(nn.Module):
    """
    在伪转换中为ReLU和ClipQuan提供包装，用于监控其输出
    """
    def __init__(self, module):
        super(Hookoutput, self).__init__()
        self.activation = 0.
        self.operation = module

    def forward(self, x):
        output = self.operation(x)
        self.activation = output.detach()
        return output


class Scale(nn.Module):
    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.register_buffer('scale', scale)

    def forward(self, x):
        if len(self.scale.shape) == 1:
            return self.scale.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        else:
            return self.scale * x


def reset(self):
    """
    转换的网络来自ANN，需要将新附加上的脉冲module进行reset
    判断module名称并调用各自节点的reset方法
    """
    children = list(self.named_children())
    for i, (name, child) in enumerate(children):
        if isinstance(child, (Burst, LIPool, SMaxPool)):
            child.reset()
        else:
            reset(child)


def change_sleep(self):
    children = list(self.named_children())
    for i, (name, child) in enumerate(children):
        if isinstance(child, Burst):
            child.sleep = not child.sleep
        else:
            change_sleep(child)



class CQConvertor(nn.Module):
    def __init__(self, soft_mode, lipool, gamma, pseudo_convert=False, merge=True, neg=False, **kwargs):
        super(CQConvertor, self).__init__()
        self.lipool = lipool
        self.soft_mode = soft_mode
        self.gamma = gamma
        self.pseudo_convert = pseudo_convert
        self.merge = merge
        self.neg = neg
        self.sleep_time = kwargs['sleep_time'] if 'sleep_time' in kwargs else [16, 16]

    def forward(self, model):
        model.eval()
        model = mergeConvBN(deepcopy(model)) if self.merge else deepcopy(model)
        # model = CQConvertor.change_beta(model, sleep_time=self.sleep_time)
        model = CQConvertor.replace_for_spike(model, self.lipool, self.soft_mode, self.gamma, self.neg, self.pseudo_convert)
        # self.model2 = deepcopy(model)

        model.reset = types.MethodType(reset, model)
        model.change_sleep = types.MethodType(change_sleep, model)
        return model

    @staticmethod
    def replace_for_spike(model, lipool=True, soft_mode=True, gamma=1, neg=False, pseudo_convert=True):
        children = list(model.named_children())
        for _, (name, child) in enumerate(children):
            if isinstance(child, ClipQuan):
                model._modules[name] = CQConvertor.get_replacement_cq(child, soft_mode, gamma, neg, pseudo_convert)
            elif isinstance(child, nn.MaxPool2d) and not pseudo_convert:
                model._modules[name] = LIPool(child) if lipool else SMaxPool(child)
            else:
                CQConvertor.replace_for_spike(child, lipool, soft_mode, gamma, neg, pseudo_convert)
        return model

    @staticmethod
    def change_beta(model, **kwargs):
        sleep_time = kwargs['sleep_time'] if 'sleep_time' in kwargs else [16, 16]
        children = list(model.named_children())
        for _, (name, child) in enumerate(children):
            if isinstance(child, nn.Conv2d) or isinstance(child, nn.Linear):
                if hasattr(model._modules[name], 'bias'):
                    model._modules[name].bias.data *= sleep_time[0]/sleep_time[1]
            else:
                CQConvertor.change_beta(child)
        return model

    @staticmethod
    def get_replacement_cq(child, soft_mode, gamma, neg, pseudo_convert):
        module_out = nn.Sequential()
        node = Burst(threshold=1., soft_mode=soft_mode, gamma=gamma, neg=neg)
        module_out.add_module('neuron', node) if not pseudo_convert else module_out.add_module('cq', Hookoutput(child))
        return module_out


class PercentConvertor(nn.Module):
    def __init__(self, dataloader, device, p, channelnorm, soft_mode, lipool, gamma, pseudo_convert=False, merge=True, neg=False, **kwargs):
        super(PercentConvertor, self).__init__()
        self.dataloader = dataloader
        self.device = device
        self.p = p
        self.channelnorm = channelnorm

        self.lipool = lipool
        self.soft_mode = soft_mode
        self.gamma = gamma
        self.pseudo_convert = pseudo_convert
        self.merge = merge
        self.neg = neg

        self.batch_num = kwargs['batch_num'] if 'batch_num' in kwargs else 1

    def forward(self, model):
        model.eval()
        model = PercentConvertor.register_hook(model, self.p, self.channelnorm, self.gamma)
        model = PercentConvertor.get_percentile(model, self.dataloader, self.device, batch_num=self.batch_num)
        model = mergeConvBN(model) if self.merge else model
        model = PercentConvertor.replace_for_spike(model, self.lipool, self.soft_mode, self.gamma, self.neg, self.pseudo_convert)
        model.reset = types.MethodType(reset, model)
        model.change_sleep = types.MethodType(change_sleep, model)
        return model

    @staticmethod
    def register_hook(model, p=0.9999, channelnorm=False, gamma=0.999):
        children = list(model.named_children())
        for _, (name, child) in enumerate(children):
            if isinstance(child, nn.ReLU):
                model._modules[name] = nn.Sequential(nn.ReLU(), HookScale(p, channelnorm, gamma))
            else:
                PercentConvertor.register_hook(child, p, channelnorm, gamma)
        return model

    @staticmethod
    def get_percentile(model, dataloader, device, batch_num=1):
        for idx, (data, _) in enumerate(dataloader):
            data = data.to(device)
            if idx >= batch_num: break
            model(data)
        return model

    @staticmethod
    def replace_for_spike(model, lipool=True, soft_mode=True, gamma=1, neg=False, pseudo_convert=True):
        children = list(model.named_children())
        for _, (name, child) in enumerate(children):
            if isinstance(child, nn.Sequential) and len(child) == 2 and isinstance(child[0], nn.ReLU) and isinstance(child[1], HookScale):
                model._modules[name] = PercentConvertor.get_replacement(child[1].scale, soft_mode, gamma, neg, pseudo_convert)
            elif isinstance(child, nn.MaxPool2d) and not pseudo_convert:
                model._modules[name] = LIPool(child) if lipool else SMaxPool(child)
            else:
                PercentConvertor.replace_for_spike(child, lipool, soft_mode, gamma, neg, pseudo_convert)
        return model

    @staticmethod
    def get_replacement(scale, soft_mode, gamma, neg, pseudo_convert):
        module_out = nn.Sequential()
        module_out.add_module('scale1', Scale(1.0 / scale))
        node = Burst(threshold=1., soft_mode=soft_mode, gamma=gamma, neg=neg)
        module_out.add_module('neuron', node) if not pseudo_convert else module_out.add_module('neuron', Hookoutput(nn.ReLU(True)))
        module_out.add_module('scale', Scale(scale))

        return module_out
