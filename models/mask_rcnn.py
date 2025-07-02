import torch
import torchvision
from torchvision.models.detection import (
    maskrcnn_resnet50_fpn,
    MaskRCNN_ResNet50_FPN_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


class MaskRCNN(torch.nn.Module):
    def __init__(self, num_classes, weights_path=None):
        super(MaskRCNN, self).__init__()
        if weights_path:
            # 使用指定路径的权重
            self.model = maskrcnn_resnet50_fpn()
            self.model.load_state_dict(torch.load(weights_path))
        else:
            # 使用默认下载的权重
            weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
            self.model = maskrcnn_resnet50_fpn(weights=weights)

        # 获取分类器的输入特征数
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features

        # 替换分类器
        if num_classes != 91:  # 如果不是COCO的类别数，才替换分类器
            self.model.roi_heads.box_predictor = FastRCNNPredictor(
                in_features, num_classes
            )

            # 获取掩码分类器的输入特征数
            in_features_mask = (
                self.model.roi_heads.mask_predictor.conv5_mask.in_channels
            )
            hidden_layer = 256

            # 替换掩码预测器
            self.model.roi_heads.mask_predictor = MaskRCNNPredictor(
                in_features_mask, hidden_layer, num_classes
            )

    def forward(self, images, targets=None):
        return self.model(images, targets)
