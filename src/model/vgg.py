'''
  @Author       : Tianlong Lee
  @Date         : 2022-06-03 01:13:31
  @LastEditTime : 2022-06-03 01:13:58
  @FilePath     : /cybord/cybord/models/vgg.py
  @Description  :
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


vgg11_arch = ((1,3,64), (1,64,128), (2,128,256), (2,256,512), (2,512,512))
vgg13_arch = ((2,3,64), (2,64,128), (2,128,256), (2,256,512), (2,512,512))
vgg16_arch = ((2,3,64), (2,64,128), (3,128,256), (3,256,512), (3,512,512))
vgg19_arch = ((2,3,64), (2,64,128), (4,128,256), (4,256,512), (4,512,512))


def vgg_block(num_convs, in_channels, out_channels, pool='max'):
    blk = []
    for i in range(num_convs):
        if i == 0:
            blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        else:
            blk.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        blk.append(nn.BatchNorm2d(out_channels))
        blk.append(nn.ReLU(inplace=True))
    if pool=='max':
        blk.append(nn.MaxPool2d(2, 2))
    elif pool=='avg':
        blk.append(nn.AvgPool2d(2, 2))
    else:
        NameError('only for max or avg mode')
    return nn.Sequential(*blk)


class VGG16(nn.Module):
    def __init__(self, pool='max', arch=vgg16_arch):
        super(VGG16, self).__init__()
        blks = []
        for i, (num_convs, in_channels, out_channels) in enumerate(arch):
            block = vgg_block(num_convs, in_channels, out_channels, pool)
            for j in range(num_convs * 3 + 1):
                blks.append(block[j])

        self.features = nn.Sequential(*blks)
        self.fc = nn.Linear(512, 1000)
        # self.fc = nn.Linear(25088, 1000)
        # self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        # self.fc = nn.Sequential(
        #     nn.Linear(25088, 4096),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(),
        #     nn.Linear(4096, 4096),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(),
        #     nn.Linear(4096, 1000))

    def forward(self, X):
        feature = self.features(X)
        if hasattr(self, 'avgpool'): feature = self.avgpool(feature)
        output = self.fc(feature.view(feature.shape[0], -1))

        # feature = self.avgpool(feature)
        # output = self.fc(feature.view(feature.shape[0], -1))
        return output
