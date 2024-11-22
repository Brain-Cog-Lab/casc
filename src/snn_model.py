import torch
import torch.nn as nn
from encoder import Encoder
from neuron import *
from copy import deepcopy


class SpikeNet(nn.Module):
    def __init__(self,
                 ori_net,
                 T=4,
                 encode_type='direct',
                 neuron=None,
                 in_channels=3,
                 **kwargs):
        super(SpikeNet, self).__init__()

        self.encoder = Encoder(T, encode_type=encode_type)
        self.neuron = neuron
        self.T = T
        self.inchannels = in_channels

        self.net = self.replace_ReLU_with_neuron(ori_net, neuron)
        SpikeNet.change_first_channel(self.net, in_channels)

    def forward(self, inputs):
        if len(inputs.shape) == 4:
            inputs = self.encoder(inputs)
        else:
            inputs = inputs.permute(1, 0, 2, 3, 4)

        self.reset()

        outputs = []
        for t in range(self.T):
            outputs.append(self.net(inputs[t]))

        outputs = torch.stack(outputs, dim=0)
        if self.training:
            self.outputs = outputs.detach()

        if hasattr(self, 'att'):
            return (self.att * outputs).mean(0)
        else:
            return outputs.mean(0)


    def reset(self):
        for ind, layer in enumerate(self.net.modules()):
            if hasattr(layer, 'reset') and ind != 0:
                layer.reset()


    @staticmethod
    def change_first_channel(model, in_channels=2):
        children = list(model.named_children())
        for _, (name, child) in enumerate(children):
            if isinstance(child, nn.Conv2d):
                i, o, k, s, p, d = child.in_channels, child.out_channels, child.kernel_size, child.stride, child.padding, child.dilation
                b = True if child.bias is not None else False
                if in_channels != i:
                    model._modules[name] = nn.Conv2d(in_channels, o, k, s, p, d, bias=b)
                break
            else:
                SpikeNet.change_first_channel(child, in_channels)
        return model


    @staticmethod
    def replace_ReLU_with_neuron(model, neuron):
        for name, module in model._modules.items():
            if hasattr(module, "_modules"):
                model._modules[name] = SpikeNet.replace_ReLU_with_neuron(module, neuron)
            if isinstance(module, nn.ReLU):
                    model._modules[name] = deepcopy(neuron)
        return model


if __name__ == '__main__':
    from model import *
    net = VGGNet()
    snn = SpikeNet(net,
                   T=4,
                   encode_type='direct',
                   neuron='LIF',
                   act_func='PiecewiseLinearGrad',
                   threshold=0.5,
                   tau=2,
                   soft_reset=False
                   )