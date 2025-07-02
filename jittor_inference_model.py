from typing import Dict, Tuple, List

import jittor as jt
from jittor.nn import interpolate
from jittor import nn
import torch
import numpy as np
from models.mask_rcnn import MaskRCNN
from models import JittorBackbone, FeaturePyramidNetwork, JittorRPN, JittorROIHeads
from utils import ImageList, jittor_resize_image


class JittorMaskRCNN(nn.Module):
    def __init__(self, weights_path=None):
        super(JittorMaskRCNN, self).__init__()

        # 保存权重路径
        self.weights_path = weights_path

        # 创建模型组件
        self.backbone = JittorBackbone()
        self.fpn = FeaturePyramidNetwork()
        self.rpn = JittorRPN()
        self.roi_heads = JittorROIHeads()

        # 如果提供了权重路径，加载权重
        if weights_path:
            self._load_weights(weights_path)

    def _load_weights(self, torch_model):
        """将PyTorch模型权重转换为Jittor格式"""
        print("\n=== 开始加载权重 ===")

        try:
            # 获取PyTorch模型的状态字典
            torch_state_dict = torch.load(self.weights_path)
            print(f"成功加载PyTorch权重文件: {self.weights_path}")

            # 获取并打印所有可用的权重名称
            print("\nPyTorch权重键值:")
            for key in torch_state_dict.keys():
                print(f"  {key}")

            weight_mapping = {
                # Backbone映射
                "backbone.body.conv1.weight": "backbone.conv1.weight",
                "backbone.body.bn1.weight": "backbone.bn1.weight",
                "backbone.body.bn1.bias": "backbone.bn1.bias",
                "backbone.body.bn1.running_mean": "backbone.bn1.running_mean",
                "backbone.body.bn1.running_var": "backbone.bn1.running_var",
                # Layer映射 - 移除'body'层级
                "backbone.body.layer1": "backbone.layer1",
                "backbone.body.layer2": "backbone.layer2",
                "backbone.body.layer3": "backbone.layer3",
                "backbone.body.layer4": "backbone.layer4",
                # FPN映射保持不变
                "backbone.fpn.inner_blocks.0": "fpn.lateral_convs.0",
                "backbone.fpn.inner_blocks.1": "fpn.lateral_convs.1",
                "backbone.fpn.inner_blocks.2": "fpn.lateral_convs.2",
                "backbone.fpn.inner_blocks.3": "fpn.lateral_convs.3",
                "backbone.fpn.layer_blocks.0": "fpn.fpn_convs.0",
                "backbone.fpn.layer_blocks.1": "fpn.fpn_convs.1",
                "backbone.fpn.layer_blocks.2": "fpn.fpn_convs.2",
                "backbone.fpn.layer_blocks.3": "fpn.fpn_convs.3",
                # 其他映射保持不变
                "rpn.head.conv": "rpn.conv",
                "rpn.head.cls_logits": "rpn.cls_logits",
                "rpn.head.bbox_pred": "rpn.bbox_pred",
                "roi_heads.box_head.fc6": "roi_heads.box_head.0",
                "roi_heads.box_head.fc7": "roi_heads.box_head.2",
                "roi_heads.box_predictor.cls_score": "roi_heads.box_predictor.0",
                "roi_heads.box_predictor.bbox_pred": "roi_heads.box_predictor.1",
                "roi_heads.mask_head.mask_fcn1": "roi_heads.mask_head.0",
                "roi_heads.mask_head.mask_fcn2": "roi_heads.mask_head.2",
                "roi_heads.mask_head.mask_fcn3": "roi_heads.mask_head.4",
                "roi_heads.mask_head.mask_fcn4": "roi_heads.mask_head.6",
                "roi_heads.mask_predictor.conv5_mask": "roi_heads.mask_predictor",
            }

            # 自动生成权重映射
            successful_mappings = 0
            failed_mappings = 0

            for jt_name, jt_param in self.named_parameters():
                # 尝试找到对应的PyTorch权重名称
                torch_name = self._find_matching_torch_name(
                    jt_name, torch_state_dict.keys()
                )

                if torch_name and torch_name in torch_state_dict:
                    try:
                        # 获取PyTorch参数
                        torch_param = torch_state_dict[torch_name].cpu().numpy()

                        # 检查形状是否匹配
                        if jt_param.shape == torch_param.shape:
                            # 更新参数
                            jt_param.update(jt.array(torch_param).float32())
                            successful_mappings += 1
                            print(
                                f"成功映射: {torch_name}{torch_param.shape} -> {jt_name}{jt_param.shape}"
                            )
                        else:
                            failed_mappings += 1
                            print(
                                f"形状不匹配: {torch_name} ({torch_param.shape}) -> {jt_name} ({jt_param.shape})"
                            )
                    except Exception as e:
                        failed_mappings += 1
                        print(f"参数转换失败: {torch_name} -> {jt_name}, 错误: {str(e)}")
                else:
                    failed_mappings += 1
                    print(f"未找到匹配的PyTorch权重: {jt_name}")

            print(f"\n权重加载完成: 成功 {successful_mappings}, 失败 {failed_mappings}")

        except Exception as e:
            print(f"权重加载过程中发生错误: {str(e)}")
            raise

    def _find_matching_torch_name(self, jt_name, torch_keys):
        """查找匹配的PyTorch权重名称"""

        # 转换Jittor名称到PyTorch名称
        def convert_name(name):
            # 处理backbone的特殊情况
            if name.startswith("backbone."):
                return "backbone.body." + name[len("backbone.") :]

            # 处理fpn的特殊情况
            if name.startswith("fpn."):
                name = "backbone.fpn." + name[len("fpn.") :]
                name = name.replace("lateral_convs", "inner_blocks")
                name = name.replace("fpn_convs", "layer_blocks")
                return name

            # 处理rpn的特殊情况
            if name.startswith("rpn."):
                for suffix in ["conv", "cls_logits", "bbox_pred"]:
                    if suffix in name:
                        return name.replace(f"rpn.{suffix}", f"rpn.head.{suffix}")

            # 处理roi_heads的特殊情况
            if name.startswith("roi_heads."):
                name = name.replace("box_head.0", "box_head.fc6")
                name = name.replace("box_head.2", "box_head.fc7")
                name = name.replace("box_predictor.0", "box_predictor.cls_score")
                name = name.replace("box_predictor.1", "box_predictor.bbox_pred")
                name = name.replace("mask_head.0", "mask_head.mask_fcn1")
                name = name.replace("mask_head.2", "mask_head.mask_fcn2")
                name = name.replace("mask_head.4", "mask_head.mask_fcn3")
                name = name.replace("mask_head.6", "mask_head.mask_fcn4")
                name = name.replace("mask_predictor.0", "mask_predictor.conv5_mask")
                name = name.replace(
                    "mask_predictor.2", "mask_predictor.mask_fcn_logits"
                )
                return name

            return name

        # 转换Jittor名称
        torch_name = convert_name(jt_name)

        # 检查转换后的名称是否在PyTorch权重中
        if torch_name in torch_keys:
            return torch_name

        return None

    def execute(self, x):
        im_h, im_w = x.shape[2:]
        origin_size = (im_h, im_w)
        print(f"origin_size: {origin_size}")
        image_list = jittor_resize_image(x)
        print(f"image_list: {image_list.tensors.shape}, {image_list.image_sizes}")
        # 提取特征
        features = self.backbone(image_list.tensors)
        fpn_features = self.fpn(features)

        # RPN预测
        rpn_output, _ = self.rpn(images=image_list, features=fpn_features)

        # ROI预测
        detections = self.roi_heads(
            features=fpn_features,
            proposals=rpn_output,
            image_shapes=image_list.image_sizes,
        )

        detections = self.postprocess(
            detections, image_list.image_sizes[0], origin_size
        )

        return detections

    def preprocess_image(self, image):
        # 图像预处理
        if isinstance(image, np.ndarray):
            # 转换为Jittor张量
            image = jt.array(image)

        # 确保图像格式正确 (B, C, H, W)
        if len(image.shape) == 3:
            image = image.unsqueeze(0)

        return image

    def postprocess(
        self,
        result: Dict,
        image_shapes: Tuple[int, int],
        original_image_shapes: Tuple[int, int],
    ):
        """ """
        print(
            f"before postprocess: {result['masks'].shape}, {result['boxes'].shape}, {result['labels'].shape}, {result['scores'].shape}"
        )
        boxes = result["boxes"]
        boxes = self.resize_boxes(boxes, image_shapes, original_image_shapes)
        result["boxes"] = boxes

        masks = result["masks"]
        masks = self.paste_masks_in_image(masks, boxes, original_image_shapes)
        result["masks"] = masks

        return result

    def resize_boxes(self, boxes, original_size, new_size):
        """
        将检测框从原始图像尺寸调整到新的图像尺寸

        参数:
            boxes (jt.Var): 形状为(N, 4)的检测框张量，格式为(xmin, ymin, xmax, ymax)
            original_size (List[int]): 原始图像尺寸 [height, width]
            new_size (List[int]): 新图像尺寸 [height, width]

        返回:
            jt.Var: 调整后的检测框张量，形状为(N, 4)
        """
        # 计算高度和宽度的缩放比例
        ratio_height = jt.float32(new_size[0]) / jt.float32(original_size[0])
        ratio_width = jt.float32(new_size[1]) / jt.float32(original_size[1])

        # 分解检测框坐标
        xmin, ymin, xmax, ymax = jt.unbind(boxes, dim=1)

        # 分别调整x和y坐标
        xmin = xmin * ratio_width
        xmax = xmax * ratio_width
        ymin = ymin * ratio_height
        ymax = ymax * ratio_height

        # 重新组合检测框坐标
        return jt.stack((xmin, ymin, xmax, ymax), dim=1)

    def _debug_parameters(self):
        """打印所有参数的名称和形状，用于调试"""
        for name, param in self.named_parameters():
            print(f"Jittor parameter: {name}, shape: {param.shape}")

    def expand_masks(self, masks, padding=1):
        """
        扩展mask以增加padding

        参数:
            masks (jt.Var): shape为[N, 1, H, W]的mask张量
            padding (int): padding大小

        返回:
            jt.Var: 扩展后的masks
            float: 缩放比例
        """
        N = masks.shape[0]
        M = masks.shape[-1]
        scale = float(M + 2 * padding) / M
        padded_masks = nn.pad(masks, (padding,) * 4)
        return padded_masks, scale

    def expand_boxes(self, boxes, scale):
        """
        按比例扩展边界框

        参数:
            boxes (jt.Var): shape为[N, 4]的边界框张量
            scale (float): 缩放比例

        返回:
            jt.Var: 扩展后的边界框
        """
        w_half = (boxes[:, 2] - boxes[:, 0]) * 0.5
        h_half = (boxes[:, 3] - boxes[:, 1]) * 0.5
        x_c = (boxes[:, 2] + boxes[:, 0]) * 0.5
        y_c = (boxes[:, 3] + boxes[:, 1]) * 0.5

        w_half = w_half * scale
        h_half = h_half * scale

        boxes_exp = jt.zeros_like(boxes)
        boxes_exp[:, 0] = x_c - w_half
        boxes_exp[:, 2] = x_c + w_half
        boxes_exp[:, 1] = y_c - h_half
        boxes_exp[:, 3] = y_c + h_half
        return boxes_exp

    def paste_mask_in_image(self, mask, box, im_h, im_w):
        """
        将单个mask粘贴到指定大小的图像中

        参数:
            mask (jt.Var): 单个mask，shape为(H, W)
            box (jt.Var): 边界框坐标 [x1, y1, x2, y2]
            im_h (int): 目标图像高度
            im_w (int): 目标图像宽度

        返回:
            jt.Var: 粘贴后的mask，shape为(im_h, im_w)
        """
        TO_REMOVE = 1
        w = int(box[2] - box[0] + TO_REMOVE)
        h = int(box[3] - box[1] + TO_REMOVE)
        w = max(w, 1)
        h = max(h, 1)

        # 设置shape为[batchxCxHxW]
        mask = mask.unsqueeze(0).unsqueeze(0)  # 从(H,W)变为(1,1,H,W)

        # 调整mask大小
        mask = nn.interpolate(mask, size=(h, w), mode="bilinear", align_corners=False)
        mask = mask[0][0]  # 返回到(H,W)

        # 创建目标mask
        im_mask = jt.zeros((im_h, im_w))
        x_0 = max(int(box[0]), 0)
        x_1 = min(int(box[2] + 1), im_w)
        y_0 = max(int(box[1]), 0)
        y_1 = min(int(box[3] + 1), im_h)

        # 将调整大小后的mask复制到正确的位置
        im_mask[y_0:y_1, x_0:x_1] = mask[
            (y_0 - int(box[1])) : (y_1 - int(box[1])),
            (x_0 - int(box[0])) : (x_1 - int(box[0])),
        ]
        return im_mask

    def paste_masks_in_image(self, masks, boxes, img_shape, padding=1):
        """
        将多个mask粘贴到指定大小的图像中

        参数:
            masks (jt.Var): shape为[N, 1, H, W]的mask张量
            boxes (jt.Var): shape为[N, 4]的边界框张量
            img_shape (Tuple[int, int]): 目标图像的(高度,宽度)
            padding (int): padding大小

        返回:
            jt.Var: 所有粘贴后的masks，shape为[N, 1, im_h, im_w]
        """
        masks, scale = self.expand_masks(masks, padding=padding)
        boxes = self.expand_boxes(boxes, scale)
        boxes = boxes.int64()
        im_h, im_w = img_shape

        res = []
        for mask, box in zip(masks, boxes):
            print(f"mask: {mask.shape}, box: {box.shape}, mask[0]: {mask[0].shape}")
            res.append(self.paste_mask_in_image(mask[0], box, im_h, im_w))

        if len(res) > 0:
            ret = jt.stack(res, dim=0)[:, None]
        else:
            ret = jt.zeros((0, 1, im_h, im_w))

        return ret
