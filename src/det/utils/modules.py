import torch
import torch.nn as nn
import torch.nn.functional as F

def get_activation(name="relu", inplace=True):
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    elif name is None:
        module = nn.Identity()
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


# Basic conv layer
class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, p=0, s=1, d=1, g=1, act='relu', depthwise=False, bias=False):
        super(Conv, self).__init__()
        if depthwise:
            assert c1 == c2
            self.convs = nn.Sequential(
                nn.Conv2d(c1, c2, k, stride=s, padding=p, dilation=d, groups=c1, bias=bias),
                nn.BatchNorm2d(c2),
                get_activation(name=act),
                nn.Conv2d(c2, c2, kernel_size=1, bias=bias),
                nn.BatchNorm2d(c2),
                get_activation(name=act)
            )
        else:
            self.convs = nn.Sequential(
                nn.Conv2d(c1, c2, k, stride=s, padding=p, dilation=d, groups=g, bias=bias),
                nn.BatchNorm2d(c2),
                get_activation(name=act)
            )

    def forward(self, x):
        return self.convs(x)


# ConvBlocks
class ConvBlocks(nn.Module):
    def __init__(self, c1, c2, act='relu'):  # in_channels, inner_channels
        super().__init__()
        c_ = c2 *2
        self.convs = nn.Sequential(
            Conv(c1, c2, k=1, act=act),
            Conv(c2, c_, k=3, p=1, act=act),
            Conv(c_, c2, k=1, act=act),
            Conv(c2, c_, k=3, p=1, act=act),
            Conv(c_, c2, k=1, act=act)
        )

    def forward(self, x):
        return self.convs(x)


# Dilated Encoder
class DilatedBottleneck(nn.Module):
    def __init__(self, c, d=1, e=0.5, act='relu'):
        super(DilatedBottleneck, self).__init__()
        c_ = int(c * e)
        self.branch = nn.Sequential(
            Conv(c, c_, k=1, act=act),
            Conv(c_, c_, k=3, p=d, d=d, act=act),
            Conv(c_, c, k=1, act=act)
        )

    def forward(self, x):
        return x + self.branch(x)


class DilatedEncoder(nn.Module):
    """ DilateEncoder """
    def __init__(self, c1, c2, act='relu', dilation_list=[2, 4, 6, 8]):
        super(DilatedEncoder, self).__init__()
        self.projector = nn.Sequential(
            Conv(c1, c2, k=1, act=None),
            Conv(c2, c2, k=3, p=1, act=None)
        )
        encoders = []
        for d in dilation_list:
            encoders.append(DilatedBottleneck(c=c2, d=d, act=act))
        self.encoders = nn.Sequential(*encoders)

    def forward(self, x):
        x = self.projector(x)
        x = self.encoders(x)

        return x


class SPP(nn.Module):
    """
        Spatial Pyramid Pooling
    """
    def __init__(self):
        super(SPP, self).__init__()
        self.max1 = nn.MaxPool2d(5, 1, 2)
        self.max2 = nn.MaxPool2d(9, 1, 4)
        self.max3 = nn.MaxPool2d(13, 1, 6)

        # self.max1 = nn.AvgPool2d(5, 1, 2)
        # self.max2 = nn.AvgPool2d(9, 1, 4)
        # self.max3 = nn.AvgPool2d(13, 1, 6)

    def forward(self, x):
        # x_1 = torch.nn.functional.max_pool2d(x, 5, stride=1, padding=2)
        # x_2 = torch.nn.functional.max_pool2d(x, 9, stride=1, padding=4)
        # x_3 = torch.nn.functional.max_pool2d(x, 13, stride=1, padding=6)

        x_1 = self.max1(x)
        x_2 = self.max2(x)
        x_3 = self.max3(x)

        x = torch.cat([x, x_1, x_2, x_3], dim=1)
        return x
