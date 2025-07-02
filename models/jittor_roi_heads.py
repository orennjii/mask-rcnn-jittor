from typing import List, Dict, Optional
import torch
from sympy.stats.sampling.sample_numpy import numpy
from torchvision.ops import MultiScaleRoIAlign, boxes as box_ops
from torchvision.models.detection.rpn import RegionProposalNetwork, AnchorGenerator
import jittor as jt
from jittor import nn
from ._utils import (
    JittorBoxCoder,
    remove_small_boxes,
    torch_batched_nms,
    jittor_batched_nms,
)


class JittorMultiScaleRoIAlign(nn.Module):
    def __init__(self, output_size, sampling_ratio):
        """
        使用torch的MultiScaleRoIAlign实现ROI Align

        Args:
            output_size: 输出的大小
            sampling_ratio: 采样比率
        """
        super(JittorMultiScaleRoIAlign, self).__init__()
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        self.output_size = output_size
        self.sampling_ratio = sampling_ratio
        self.scales = []
        # 创建torch的MultiScaleRoIAlign实例
        self.torch_roi_align = MultiScaleRoIAlign(
            featmap_names=["0", "1", "2", "3"],  # 对应特征层级的名称
            output_size=output_size,
            sampling_ratio=sampling_ratio,
        )

    def execute(self, features, proposals, image_shapes) -> jt.Var:
        """
        Args:
            features: List[jt.Var] - 特征图列表
            proposals: List[jt.Var] - proposal boxes列表
            image_shapes: List[Tuple[H, W]] - 输入图像尺寸列表
        Returns:
            jt.Var - ROI Align的结果
        """

        # 1. 将jittor数据转换为torch tensor
        torch_features = {}
        for i, feat in enumerate(features):
            torch_features[str(i)] = torch.from_numpy(feat.numpy())

        torch_proposals = [torch.from_numpy(p.numpy()) for p in proposals]

        # 2. 使用torch的MultiScaleRoIAlign
        with torch.no_grad():
            result = self.torch_roi_align(torch_features, torch_proposals, image_shapes)

        # 3. 将结果转回jittor.Var
        return jt.array(result.numpy())


class JittorROIHeads(nn.Module):
    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 91,
        representation_size: int = 1024,
        score_thresh: float = 0.05,
        nms_thresh: float = 0.5,
        box_detections_per_img: int = 100,
    ) -> None:
        super(JittorROIHeads, self).__init__()
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.box_detection_per_ing = box_detections_per_img
        self.box_coder = JittorBoxCoder(weights=(10.0, 10.0, 5.0, 5.0))

        # ROI Pooling层
        self.box_roi_pool = JittorMultiScaleRoIAlign(
            output_size=(7, 7), sampling_ratio=2
        )
        self.mask_roi_pool = JittorMultiScaleRoIAlign(
            output_size=(14, 14), sampling_ratio=2
        )

        # Box Head
        self.box_head = nn.Sequential(
            nn.Linear(in_channels * 7 * 7, representation_size),
            nn.ReLU(),
            nn.Linear(representation_size, representation_size),
            nn.ReLU(),
        )

        # Box Predictor
        self.box_predictor = nn.Sequential(
            nn.Linear(representation_size, num_classes),  # 分类
            nn.Linear(representation_size, num_classes * 4),  # 边界框回归
        )

        # Mask Head - 4个卷积层
        self.mask_head = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),  # mask_fcn1
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # mask_fcn2
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # mask_fcn3
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # mask_fcn4
            nn.ReLU(),
        )

        # Mask Predictor
        self.mask_predictor = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 256, kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0),
        )

    def _postprocess_detections(
        self,
        class_logits: jt.Var,
        box_regression: jt.Var,
        proposals: List[jt.Var],
        image_shapes: List[tuple],
    ):
        """
        处理类别分类(class_logics)，和边界框回归(box_regression)的原始预测结果
        处理成最终的检测输出(包括边界框、分数、类别标签)
        Args:
            class_logits: jt.Var - 类别分类预测
            box_regression: jt.Var - 边界框回归预测
            proposals: List[jt.Var] - RPN生成的proposals列表
            image_shapes: List[tuple] - 原始图像尺寸列表
        Returns:

        """
        num_classes = class_logits.shape[-1]

        boxes_per_image = [boxes.shape[0] for boxes in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals[0])

        pred_scores = jt.nn.softmax(class_logits, dim=-1)

        pred_boxes_list = jt.split(pred_boxes, boxes_per_image)
        pred_scores_list = jt.split(pred_scores, boxes_per_image)

        all_boxes = []
        all_labels = []
        all_scores = []
        for boxes, scores, image_shapes in zip(
            pred_boxes_list, pred_scores_list, image_shapes
        ):
            boxes = self._clip_boxes_to_image(boxes, image_shapes)
            # 创建labels为每一个prediction
            labels = jt.arange(num_classes)
            labels = jt.view(labels, (1, -1)).expand(scores.shape[0], -1)

            # 移除背景类别
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]
            print(
                f"boxes: {boxes.shape}, scores: {scores.shape}, labels: {labels.shape}"
            )

            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            # 移除低分数的预测
            inds = jt.where(scores > self.score_thresh)[0]
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]
            print(
                f"boxes: {boxes.shape}, scores: {scores.shape}, labels: {labels.shape}"
            )

            # 移除空的预测
            keep = remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            print(
                f"boxes: {boxes.shape}, scores: {scores.shape}, labels: {labels.shape}"
            )

            # NMS
            keep = torch_batched_nms(boxes, scores, labels, self.nms_thresh)
            keep = keep[: self.box_detection_per_ing]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            print(
                f"boxes: {boxes.shape}, scores: {scores.shape}, labels: {labels.shape}"
            )

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        return all_boxes, all_scores, all_labels

    def execute(
        self,
        features: List[jt.Var],
        proposals: List[jt.Var],
        image_shapes: List[tuple],
        targets: Optional[Dict[str, jt.Var]] = None,
    ) -> Dict[str, jt.Var]:
        """
        Args:
            features: List[jt.Var] - FPN特征图列表
            proposals: List[jt.Var] - RPN生成的proposals列表，每个元素shape为[N, 4]
            image_shapes: List[tuple] - 原始图像尺寸列表
            targets: Optional[Dict[str, jt.Var]] - 训练时的目标信息
        """
        print("\nROI Heads Input Shapes:")
        print(f"Features shapes: {[f.shape for f in features]}")
        print(f"Proposals shapes: {[p.shape for p in proposals]}")

        # 1. ROI Pooling
        box_features = self.box_roi_pool(features, proposals, image_shapes)
        print(f"ROI pooled features shape: {box_features.shape}")

        # 2. Box Head
        # 展平特征用于全连接层
        box_features = box_features.reshape(box_features.shape[0], -1)
        print(f"Flattened box features shape: {box_features.shape}")

        # 通过box head网络
        box_features = self.box_head(box_features)

        # 3. Box Prediction
        # 分类预测
        class_logits = self.box_predictor[0](box_features)
        # 边界框回归预测
        box_regression = self.box_predictor[1](box_features)

        print(f"Class logits shape: {class_logits.shape}")
        print(f"Box regression shape: {box_regression.shape}")

        boxes, scores, labels = self._postprocess_detections(
            class_logits=class_logits,
            box_regression=box_regression,
            proposals=proposals,
            image_shapes=image_shapes,
        )
        print(
            f"Boxes shape: {boxes[0].shape}, Scores shape: {scores[0].shape}, Labels shape: {labels[0].shape}"
        )

        # 4. Mask Head
        mask_proposals = boxes

        mask_features = self.mask_roi_pool(features, mask_proposals, image_shapes)
        mask_features = self.mask_head(mask_features)
        mask_logits = self.mask_predictor(mask_features)

        mask_probs = self.maskrcnn_inference(mask_logits, labels[0])

        print(f"Mask features shape: {mask_features.shape}")
        print(f"Mask logits shape: {mask_logits.shape}")
        print(f"labels shape: {labels[0].shape}")

        return {
            "boxes": boxes[0],
            "labels": labels[0],
            "scores": scores[0],
            "masks": mask_probs,
        }

    def _clip_boxes_to_image(self, boxes, image_shape):
        """确保boxes在图像范围内"""
        h, w = image_shape
        boxes[:, 0] = jt.clamp(boxes[:, 0], min_v=0, max_v=w)
        boxes[:, 1] = jt.clamp(boxes[:, 1], min_v=0, max_v=h)
        boxes[:, 2] = jt.clamp(boxes[:, 2], min_v=0, max_v=w)
        boxes[:, 3] = jt.clamp(boxes[:, 3], min_v=0, max_v=h)
        return boxes

    def maskrcnn_inference(self, x, labels):
        """
        从CNN结果中后处理masks，通过选择对应最大概率类别的mask

        Args:
            x (jt.Var): mask logits [N, num_classes, H, W]
            labels (jt.Var): 预测的类别标签 [N]

        Returns:
            jt.Var: 处理后的mask概率
        """
        # 计算mask概率
        mask_prob = x.sigmoid()

        # 选择对应预测类别的masks
        num_masks = x.shape[0]

        # 创建索引
        index = jt.arange(num_masks)

        # 选择每个预测框对应类别的mask
        mask_prob = mask_prob[index, labels]

        # 添加channel维度
        mask_prob = mask_prob.unsqueeze(1)

        return mask_prob

    def assign_targets_to_proposals(self, proposals, gt_boxes, gt_labels):
        """为proposals分配ground truth标签

        Args:
            proposals: List[jt.Var] - RPN生成的proposals
            gt_boxes: List[jt.Var] - ground truth boxes
            gt_labels: List[jt.Var] - ground truth labels

        Returns:
            List[jt.Var], List[jt.Var] - 匹配的标签和回归目标
        """
        matched_idxs = []
        labels = []

        for proposals_in_image, gt_boxes_in_image, gt_labels_in_image in zip(
            proposals, gt_boxes, gt_labels
        ):
            if gt_boxes_in_image.numel() == 0:
                # 背景类
                labels_in_image = jt.zeros(
                    (proposals_in_image.shape[0],), dtype=jt.int32
                )
                matched_idxs_in_image = jt.zeros(
                    (proposals_in_image.shape[0],), dtype=jt.int32
                )
            else:
                # 计算IoU
                match_quality_matrix = box_ops.box_iou(
                    gt_boxes_in_image, proposals_in_image
                )

                # 使用最大IoU匹配
                matched_vals, matches = match_quality_matrix.max(dim=0)

                # 根据IoU阈值分配标签
                labels_in_image = gt_labels_in_image[matches]

                # 将低IoU样本标记为背景
                bg_inds = matched_vals < 0.5
                labels_in_image[bg_inds] = 0

                # 将高IoU样本标记为前景
                fg_inds = matched_vals >= 0.5
                labels_in_image[fg_inds] = gt_labels_in_image[matches[fg_inds]]

                matched_idxs_in_image = matches

            matched_idxs.append(matched_idxs_in_image)
            labels.append(labels_in_image)

        return matched_idxs, labels

    def subsample(self, labels):
        """对正负样本进行采样,保持固定比例

        Args:
            labels: List[jt.Var] - 标签列表

        Returns:
            List[jt.Var] - 采样后的标签
        """
        sampled_pos_inds, sampled_neg_inds = [], []
        for labels_per_image in labels:
            pos_inds = jt.where(labels_per_image > 0)[0]
            neg_inds = jt.where(labels_per_image == 0)[0]

            # 采样正样本
            num_pos = min(pos_inds.shape[0], 512 // 4)
            perm = jt.randperm(pos_inds.shape[0])[:num_pos]
            pos_inds = pos_inds[perm]

            # 采样负样本
            num_neg = min(neg_inds.shape[0], 512 - num_pos)
            perm = jt.randperm(neg_inds.shape[0])[:num_neg]
            neg_inds = neg_inds[perm]

            sampled_pos_inds.append(pos_inds)
            sampled_neg_inds.append(neg_inds)

        return sampled_pos_inds, sampled_neg_inds

    def compute_loss(self, class_logits, box_regression, mask_logits, targets):
        """计算ROI head的损失

        Args:
            class_logits: jt.Var - 分类预测 [N, num_classes]
            box_regression: jt.Var - 边界框回归预测 [N, num_classes * 4]
            mask_logits: jt.Var - mask预测 [N, num_classes, 28, 28]
            targets: Dict - 训练目标
                - boxes: List[jt.Var]
                - labels: List[jt.Var]
                - masks: List[jt.Var]

        Returns:
            Dict[str, jt.Var] - 损失字典
        """
        # 为proposals分配targets
        matched_idxs, labels = self.assign_targets_to_proposals(
            proposals=targets["proposals"],
            gt_boxes=targets["boxes"],
            gt_labels=targets["labels"],
        )

        # 采样
        sampled_pos_inds, sampled_neg_inds = self.subsample(labels)

        # 合并正负样本索引
        sampled_inds = []
        for pos_inds, neg_inds in zip(sampled_pos_inds, sampled_neg_inds):
            sampled_inds.append(jt.concat([pos_inds, neg_inds]))

        # 分类损失
        classification_loss = jt.nn.cross_entropy_loss(
            class_logits, labels, reduction="mean"
        )

        # 只使用正样本计算box regression损失
        box_loss = jt.nn.smooth_l1_loss(
            box_regression[sampled_pos_inds],
            targets["boxes"][sampled_pos_inds],
            reduction="mean",
        )

        # Mask损失 - 只在正样本上计算
        mask_loss = jt.nn.binary_cross_entropy_with_logits(
            mask_logits[sampled_pos_inds],
            targets["masks"][sampled_pos_inds],
            reduction="mean",
        )

        losses = {
            "loss_classifier": classification_loss,
            "loss_box_reg": box_loss,
            "loss_mask": mask_loss,
        }

        return losses
