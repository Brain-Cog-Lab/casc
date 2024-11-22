import torch.nn as nn
import torch.nn.functional as F

resnet18 = ((2, 64, 64), (2, 64, 128), (2, 128, 256), (2, 256, 512))
resnet20 = ((2, 64, 64), (2, 64, 128), (2, 128, 256), (2, 256, 512))
resnet34 = ((3, 64, 64), (4, 64, 128), (6, 128, 256), (3, 256, 512))
resnet50 = ((3, 64, 64, 256), (4, 256, 128, 512), (6, 512, 256, 1024), (3, 1024, 512, 2048))
resnet101 = ((3, 64, 64, 256), (4, 256, 128, 512), (23, 512, 256, 1024), (3, 1024, 512, 2048))
resnet152 = ((3, 64, 64, 256), (8, 256, 128, 512), (36, 512, 256, 1024), (3, 1024, 512, 2048))


class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(BasicBlock, self).__init__()
        stride = 2 if in_channel != out_channel else 1
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.conv1 = nn.Conv2d(in_channel, out_channel, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channel, eps=1e-05, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel, eps=1e-05, momentum=0.1)

        if in_channel != out_channel:
            self.downsample = nn.Sequential(
                nn.BatchNorm2d(in_channel, eps=1e-05, momentum=0.1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channel, out_channel, 1, 2, bias=False))

        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, X):
        identity = X
        Y = self.conv1(self.relu(self.bn1(X)))
        Y = self.conv2(self.relu2(self.bn2(Y)))
        if self.in_channel != self.out_channel:
            identity = self.downsample(X)
        return Y + identity


class ResNet18v2(nn.Module):
    def __init__(self, arch=resnet18):
        super(ResNet18v2, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)

        layers = []
        for i, (num_residual, in_channel, out_channel) in enumerate(arch):
            blk = []
            for j in range(num_residual):
                if j == 0:
                    blk.append(BasicBlock(in_channel, out_channel))
                else:
                    blk.append(BasicBlock(out_channel, out_channel))
            layers.append(nn.Sequential(*blk))

        self.layer1 = layers[0]
        self.layer2 = layers[1]
        self.layer3 = layers[2]
        self.layer4 = layers[3]
        self.bn = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 1000)

    def forward(self, X):
        X = self.conv1(X)
        X = self.layer4(self.layer3(self.layer2(self.layer1(X))))
        X = self.bn(X)
        X = self.avgpool(X)
        out = self.fc(X.view(X.shape[0], -1))
        return out


class ResNet20v2(nn.Module):
    def __init__(self, arch=resnet18):
        super(ResNet20v2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), nn.BatchNorm2d(64, eps=1e-05, momentum=0.1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64, eps=1e-05, momentum=0.1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1))

        layers = []
        for i, (num_residual, in_channel, out_channel) in enumerate(arch):
            blk = []
            for j in range(num_residual):
                if j == 0:
                    blk.append(BasicBlock(in_channel, out_channel))
                else:
                    blk.append(BasicBlock(out_channel, out_channel))
            layers.append(nn.Sequential(*blk))

        self.layer1 = layers[0]
        self.layer2 = layers[1]
        self.layer3 = layers[2]
        self.layer4 = layers[3]

        self.bn = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 100)

    def forward(self, X):
        X = self.conv1(X)
        X = self.layer4(self.layer3(self.layer2(self.layer1(X))))
        X = self.bn(X)
        X = self.pool(X)
        out = self.fc(X.view(X.shape[0], -1))
        return out


class ResNet34v2(nn.Module):
    def __init__(self, arch=resnet34):
        super(ResNet34v2, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)

        layers = []
        for i, (num_residual, in_channel, out_channel) in enumerate(arch):
            blk = []
            for j in range(num_residual):
                if j == 0:
                    blk.append(BasicBlock(in_channel, out_channel))
                else:
                    blk.append(BasicBlock(out_channel, out_channel))
            layers.append(nn.Sequential(*blk))

        self.layer1 = layers[0]
        self.layer2 = layers[1]
        self.layer3 = layers[2]
        self.layer4 = layers[3]
        self.bn = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 1000)

    def forward(self, X):
        X = self.conv1(X)
        X = self.layer4(self.layer3(self.layer2(self.layer1(X))))
        X = self.bn(X)
        X = self.avgpool(X)
        out = self.fc(X.view(X.shape[0], -1))
        return out


class Bottleneck(nn.Module):
    def __init__(self, in_channel, hidden_channel, out_channel, first=False):
        super(Bottleneck, self).__init__()
        if in_channel != hidden_channel:
            stride = 2
        else:
            stride = 1
        self.first = first
        if first:
            self.conv1 = nn.Conv2d(in_channel, hidden_channel, 1, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(in_channel, eps=1e-05, momentum=0.1)
            self.relu1 = nn.ReLU(inplace=True)
            if in_channel == hidden_channel:
                self.conv2 = nn.Conv2d(hidden_channel, hidden_channel, 3, 1, 1, bias=False)
            else:
                self.conv2 = nn.Conv2d(hidden_channel, hidden_channel, 3, stride, 1, bias=False)
            self.bn2 = nn.BatchNorm2d(hidden_channel, eps=1e-05, momentum=0.1)
            self.relu2 = nn.ReLU(inplace=True)
            self.conv3 = nn.Conv2d(hidden_channel, out_channel, 1, 1, bias=False)
            self.bn3 = nn.BatchNorm2d(hidden_channel, eps=1e-05, momentum=0.1)
        else:
            self.conv1 = nn.Conv2d(out_channel, hidden_channel, 1, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channel, eps=1e-05, momentum=0.1)
            self.relu1 = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(hidden_channel, hidden_channel, 3, 1, 1, bias=False)
            self.bn2 = nn.BatchNorm2d(hidden_channel, eps=1e-05, momentum=0.1)
            self.relu2 = nn.ReLU(inplace=True)
            self.conv3 = nn.Conv2d(hidden_channel, out_channel, 1, 1, bias=False)
            self.bn3 = nn.BatchNorm2d(hidden_channel, eps=1e-05, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        if first:
            if in_channel != hidden_channel:
                self.downsample = nn.Sequential(
                    nn.BatchNorm2d(in_channel, eps=1e-05, momentum=0.1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channel, out_channel, 1, 2, bias=False))
            else:
                self.downsample = nn.Sequential(
                    nn.BatchNorm2d(in_channel, eps=1e-05, momentum=0.1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channel, out_channel, 1, 1, bias=False))

    def forward(self, X):
        identity = X
        Y = self.conv1(self.relu1(self.bn1(X)))
        Y = self.conv2(self.relu2(self.bn2(Y)))
        Y = self.conv3(self.relu(self.bn3(Y)))
        if self.first:
            identity = self.downsample(X)
        return F.relu(Y + identity)


resnet50 = ((3, 64, 64, 256), (4, 256, 128, 512), (6, 512, 256, 1024), (3, 1024, 512, 2048))


class ResNet50v2(nn.Module):
    def __init__(self, arch=resnet50):
        super(ResNet50v2, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        layers = []
        for i, (num_residual, in_channel, hidden_channel, out_channel) in enumerate(arch):
            blk = []
            for j in range(num_residual):
                if j == 0:
                    blk.append(Bottleneck(in_channel, hidden_channel, out_channel, first=True))
                else:
                    blk.append(Bottleneck(in_channel, hidden_channel, out_channel))
            layers.append(nn.Sequential(*blk))

        self.layer1 = layers[0]
        self.layer2 = layers[1]
        self.layer3 = layers[2]
        self.layer4 = layers[3]
        self.bn = nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, 1000)

    def forward(self, X):
        X = self.conv1(X)
        X = self.layer4(self.layer3(self.layer2(self.layer1(X))))
        X = self.bn(X)
        X = self.avgpool(X)
        out = self.fc(X.view(X.shape[0], -1))
        return out


if __name__ == "__main__":
    net = ResNet18v2()
    import torch

    a = torch.rand(128, 3, 32, 32)
    y = net(a)
    print('done')