import jittor as jt
from jittor import nn


class FeaturePyramidNetwork(nn.Module):
    def __init__(self, in_channels_list=[256, 512, 1024, 2048], out_channels=256):
        super(FeaturePyramidNetwork, self).__init__()

        # 横向连接的1x1卷积
        self.lateral_convs = nn.ModuleList(
            [
                nn.Conv2d(in_channels, out_channels, 1)  # 1x1卷积降维
                for in_channels in in_channels_list
            ]
        )

        # 3x3卷积进行特征融合
        self.fpn_convs = nn.ModuleList(
            [
                nn.Conv2d(out_channels, out_channels, 3, padding=1)
                for _ in range(len(in_channels_list))
            ]
        )

        self.maxpool = nn.MaxPool2d(1, stride=2)  # 用于处理最底层特征

    def execute(self, inputs):
        # inputs是从backbone获取的特征列表 [C2, C3, C4, C5]
        laterals = [
            lateral_conv(inputs[i]) for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # 自顶向下的路径和特征融合
        for i in range(len(laterals) - 1, 0, -1):
            # 上采样
            upsampled = nn.interpolate(
                laterals[i], size=laterals[i - 1].shape[-2:], mode="nearest"
            )
            # 特征融合（加法）
            laterals[i - 1] = laterals[i - 1] + upsampled

        # 3x3卷积处理融合后的特征
        outs = [
            fpn_conv(lateral) for fpn_conv, lateral in zip(self.fpn_convs, laterals)
        ]
        outs.append(self.maxpool(laterals[-1]))  # 添加最底层特征

        return tuple(outs)  # [P2, P3, P4, P5, P6]
