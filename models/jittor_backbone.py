import jittor as jt
from jittor import nn
from typing import List


class JittorBackbone(nn.Module):
    def __init__(self):
        super(JittorBackbone, self).__init__()
        # 创建基础卷积层，注意设置bias=False
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 创建ResNet的四个阶段
        self.layer1 = self._make_layer(64, 256, blocks=3, stride=1)  # C2
        self.layer2 = self._make_layer(256, 512, blocks=4, stride=2)  # C3
        self.layer3 = self._make_layer(512, 1024, blocks=6, stride=2)  # C4
        self.layer4 = self._make_layer(1024, 2048, blocks=3, stride=2)  # C5

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []

        # 第一个block可能需要downsample
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        # 添加第一个block
        layers.append(Bottleneck(in_channels, out_channels // 4, stride, downsample))

        # 添加剩余的blocks
        for _ in range(1, blocks):
            layers.append(Bottleneck(out_channels, out_channels // 4))

        return nn.Sequential(*layers)

    def execute(self, x) -> List[jt.Var]:
        # 初始层
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 获取四个阶段的特征
        c2 = self.layer1(x)  # stride 4 relative to input
        c3 = self.layer2(c2)  # stride 8
        c4 = self.layer3(c3)  # stride 16
        c5 = self.layer4(c4)  # stride 32

        return [c2, c3, c4, c5]


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        # 1x1 降维，注意设置bias=False
        self.conv1 = nn.Conv2d(in_channels, channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)

        # 3x3 卷积，注意设置bias=False
        self.conv2 = nn.Conv2d(
            channels, channels, 3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(channels)

        # 1x1 升维，注意设置bias=False
        self.conv3 = nn.Conv2d(channels, channels * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels * self.expansion)

        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def execute(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
