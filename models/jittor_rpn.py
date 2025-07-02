from typing import List, Tuple, Dict
import jittor as jt
from jittor import nn
from ._utils import (
    JittorBoxCoder,
    remove_small_boxes,
    jittor_batched_nms,
    torch_batched_nms,
)


class JittorRPNHead(nn.Module):
    def __init__(self, in_channels: int = 256, num_anchors: int = 3):
        super(JittorRPNHead, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )
        self.relu = nn.ReLU()
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d(
            in_channels, num_anchors * 4, kernel_size=1, stride=1
        )

    def execute(self, x: List[jt.Var]) -> Tuple[List[jt.Var], List[jt.Var]]:
        logits = []
        bbox_reg = []
        for feature in x:
            t = self.relu(self.conv(feature))
            logits.append(self.cls_logits(t))
            bbox_reg.append(self.bbox_pred(t))

        return logits, bbox_reg


class JittorAnchorGenerator(nn.Module):
    def __init__(
        self,
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),),
        strides=(4, 8, 16, 32, 64),
    ):
        super().__init__()
        if not isinstance(sizes[0], (list, tuple)):
            sizes = tuple((s,) for s in sizes)
        if not isinstance(aspect_ratios[0], (list, tuple)):
            aspect_ratios = (aspect_ratios,) * len(sizes)

        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.strides = strides
        self.cell_anchors = None
        self._cache = {}

    def generate_anchors(self, scales, aspect_ratios, dtype=jt.float32):
        """计算基准anchors"""
        scales = jt.array(scales, dtype=dtype)
        aspect_ratios = jt.array(aspect_ratios, dtype=dtype)
        h_ratios = jt.sqrt(aspect_ratios)
        w_ratios = 1 / h_ratios

        ws = (w_ratios[:, None] * scales[None, :]).view(-1)
        hs = (h_ratios[:, None] * scales[None, :]).view(-1)

        base_anchors = jt.stack([-ws, -hs, ws, hs], dim=1) / 2
        return base_anchors.round()

    def set_cell_anchors(self, dtype):
        """生成每个特征层的基准anchors"""
        if self.cell_anchors is not None:
            return

        cell_anchors = [
            self.generate_anchors(sizes, aspect_ratios, dtype)
            for sizes, aspect_ratios in zip(self.sizes, self.aspect_ratios)
        ]
        self.cell_anchors = cell_anchors

    def grid_anchors(self, grid_sizes, strides):
        """为每个特征图生成anchors网格"""
        anchors = []
        for size, stride, base_anchors in zip(grid_sizes, strides, self.cell_anchors):
            grid_height, grid_width = size
            stride_height, stride_width = stride

            # 生成网格中心点
            shifts_x = jt.arange(0, grid_width, dtype=jt.float32) * stride_width
            shifts_y = jt.arange(0, grid_height, dtype=jt.float32) * stride_height
            shift_y, shift_x = jt.meshgrid(shifts_y, shifts_x)
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)
            shifts = jt.stack((shift_x, shift_y, shift_x, shift_y), dim=1)

            # 将基准anchors应用到每个网格点
            anchors.append(
                (shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4)
            )

        return anchors

    def cached_grid_anchors(self, grid_sizes, strides):
        """缓存生成的anchors网格
        Args:
            grid_sizes: List[Tuple[int, int]] 特征图尺寸列表
            strides: List[Tuple[int, int]] 步长列表
        """
        # 将NanoVector转换为普通元组
        key = []
        for size, stride in zip(grid_sizes, strides):
            # 将每个grid_size和stride转换为元组
            size_tuple = tuple(int(x) for x in size)
            stride_tuple = tuple(int(x) for x in stride)
            key.append((size_tuple, stride_tuple))
        key = tuple(key)  # 转换为不可的元组

        if key in self._cache:
            return self._cache[key]

        anchors = self.grid_anchors(grid_sizes, strides)
        self._cache[key] = anchors
        return anchors

    def execute(self, image_list, feature_maps):
        """生成所有特征图的anchors
        Args:
            image_list: ImageList对象，包含:
                - tensors: 批量图像张量 [N, C, H, W]
                - image_sizes: 原始图像尺寸列表 [(h1, w1), (h2, w2), ...]
            feature_maps: FPN特征图列表
        Returns:
            List[jt.Var]: 所有特征图的anchors
        """
        grid_sizes = tuple(feature_map.shape[-2:] for feature_map in feature_maps)
        image_size = image_list.tensors.shape[-2:]  # 使用预处理后的图像尺寸
        dtype = feature_maps[0].dtype

        # 计算每个特征层的stride
        strides = tuple(
            (image_size[0] // g[0], image_size[1] // g[1]) for g in grid_sizes
        )

        self.set_cell_anchors(dtype)
        anchors_over_all_feature_maps = self.cached_grid_anchors(grid_sizes, strides)

        anchors = []
        # 为每张图片生成anchors
        for i in range(len(image_list.image_sizes)):
            anchors_in_image = [
                anchors_per_feature_map
                for anchors_per_feature_map in anchors_over_all_feature_maps
            ]
            anchors.append(jt.concat(anchors_in_image))

        return anchors


def permute_and_flatten(
    layer: jt.Var, N: int, A: int, C: int, H: int, W: int
) -> jt.Var:
    """
    将特征图变换为适合RPN网络的形状
    Args:
        layer: jt.Var 输入特征图
        N: int batch size
        A: int anchor数量
        C: int 通道数
        H: int 高度
        W: int 宽度
    Returns:
        jt.Var: 变换后的特征图
    """
    layer = jt.view(layer, (N, -1, C, H, W))
    layer = jt.permute(layer, (0, 3, 4, 1, 2))
    layer = jt.reshape(layer, (N, -1, C))
    return layer


def concat_box_prediction_layers(
    box_cls: List[jt.Var], box_regression: List[jt.Var]
) -> Tuple[jt.Var, jt.Var]:
    """
    将不同特征图上的预测结果连接在一起
    Args:
        box_cls: List[jt.Var] 分类预测张量列表
        box_regression: List[jt.Var] 边界框回归预测张量列表
    Returns:
        box_cls: jt.Var 连接后的分类预测张量
        box_regression: jt.Var 连接后的边界框回归预测张量
    """
    box_cls_flattened = []
    box_regression_flattened = []
    for box_cls_per_level, box_regression_per_level in zip(box_cls, box_regression):
        N, AxC, H, W = box_cls_per_level.shape
        Ax4 = box_regression_per_level.shape[1]
        A = Ax4 // 4
        C = AxC // A
        box_cls_per_level = permute_and_flatten(box_cls_per_level, N, A, C, H, W)
        box_cls_flattened.append(box_cls_per_level)
        box_regression_per_level = permute_and_flatten(
            box_regression_per_level, N, A, 4, H, W
        )
        box_regression_flattened.append(box_regression_per_level)

    box_cls = jt.flatten(jt.concat(box_cls_flattened, dim=1), 0, -2)
    box_regression = jt.concat(box_regression_flattened, dim=1).reshape(-1, 4)

    return box_cls, box_regression


class JittorRPN(jt.nn.Module):
    def __init__(
        self,
        in_channels: int = 256,
        anchor_sizes: Tuple = ((32,), (64,), (128,), (256,), (512,)),
        aspect_ratios: Tuple = ((0.5, 1.0, 2.0),),
        rpn_pre_nms_top_n_train=2000,
        rpn_pre_nms_top_n_test=1000,
        rpn_post_nms_top_n_train=2000,
        rpn_post_nms_top_n_test=1000,
        rpn_nms_thresh=0.7,
        rpn_fg_iou_thresh=0.7,
        rpn_bg_iou_thresh=0.3,
        rpn_batch_size_per_image=256,
        rpn_positive_fraction=0.5,
        rpn_score_thresh=0.0,
    ) -> None:
        super(JittorRPN, self).__init__()

        # 保存配置参数
        self.anchor_sizes = anchor_sizes
        self.aspect_ratios = aspect_ratios
        self.rpn_pre_nms_top_n = {
            True: rpn_pre_nms_top_n_train,
            False: rpn_pre_nms_top_n_test,
        }
        self.rpn_post_nms_top_n = {
            True: rpn_post_nms_top_n_train,
            False: rpn_post_nms_top_n_test,
        }
        self.rpn_nms_thresh = rpn_nms_thresh
        self.rpn_fg_iou_thresh = rpn_fg_iou_thresh
        self.rpn_bg_iou_thresh = rpn_bg_iou_thresh
        self.rpn_batch_size_per_image = rpn_batch_size_per_image
        self.rpn_positive_fraction = rpn_positive_fraction
        self.rpn_score_thresh = rpn_score_thresh
        # 网络组件
        self.box_coder = JittorBoxCoder(weights=(1.0, 1.0, 1.0, 1.0))
        num_anchors = len(aspect_ratios[0]) * len(anchor_sizes[0])
        self.head = JittorRPNHead(in_channels, num_anchors)
        self.anchor_generator = JittorAnchorGenerator(
            sizes=anchor_sizes,
            aspect_ratios=aspect_ratios * len(anchor_sizes),
            strides=(4, 8, 16, 32, 64),  # 对应FPN的不同层级
        )

    def filter_proposals(
        self,
        proposals: jt.Var,
        objectness: jt.Var,
        image_shapes: List[Tuple[int, int]],
        num_anchors_per_level: List[int],
    ) -> Tuple[List[jt.Var], List[jt.Var]]:
        """过滤和选择最好的proposals
        Args:
            proposals: jt.Var [batch_size, num_proposals, 4]
            objectness: jt.Var [batch_size, num_proposals]
            image_shapes: List[Tuple[int, int]] 图像尺寸列表
            num_anchors_per_level: List[int] 每个特征层的anchor数量
        Returns:
            final_boxes: List[jt.Var] 过滤后的边界框列表
            final_scores: List[jt.Var] 过滤后的分数列表
        """
        num_images = proposals.shape[0]

        # 不计算objectness的梯度
        objectness = objectness.stop_grad()
        objectness = jt.reshape(objectness, (num_images, -1))

        # 为每个特征层级创建标识
        levels = []
        for idx, n in enumerate(num_anchors_per_level):
            levels.append(jt.full((n,), idx, dtype="int64"))
        levels = jt.concat(levels, 0)
        levels = jt.reshape(levels, (1, -1))

        # 在NMS之前为每个层级选择top_n个框
        top_n_idx = self._get_top_n_idx(objectness, num_anchors_per_level)

        # 创建batch索引
        batch_idx = jt.arange(num_images)
        batch_idx = jt.reshape(batch_idx, (-1, 1))

        # 根据top_n_idx选择对应的proposals和scores
        objectness = objectness[batch_idx, top_n_idx]
        levels = levels[batch_idx, top_n_idx]
        proposals = proposals[batch_idx, top_n_idx, :]

        # 计算objectness概率
        objectness_prob = jt.sigmoid(objectness)

        final_boxes = []
        final_scores = []

        # 对每张图片分别处理
        print(
            f"proposals: {proposals.shape}, objectness_prob: {objectness_prob.shape}, levels: {levels.shape}, image_shapes: {image_shapes}"
        )
        for boxes, scores, lvl, img_shape in zip(
            proposals, objectness_prob, levels, image_shapes
        ):
            # 添加调试信息
            boxes = self.clip_boxes_to_image(boxes, img_shape)

            keep = remove_small_boxes(boxes, min_size=1e-3)
            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]

            # 移除低分数的框
            keep = jt.where(scores >= self.rpn_score_thresh)[0]
            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]

            keep = jittor_batched_nms(boxes, scores, lvl, self.rpn_nms_thresh)
            print(f"After NMS - boxes: {boxes[keep].shape}")

            keep = keep[: self.rpn_post_nms_top_n[self.training]]
            boxes, scores = boxes[keep], scores[keep]
            print(f"Final boxes: {boxes.shape}")

            final_boxes.append(boxes)
            final_scores.append(scores)

        return final_boxes, final_scores

    def _get_top_n_idx(self, objectness, num_anchors_per_level):
        """获取每个特征层级的top_n个anchor的索引"""
        r = []
        offset = 0
        for ob in objectness.split(num_anchors_per_level, dim=1):
            num_anchors = ob.shape[1]
            pre_nms_top_n = min(self.rpn_pre_nms_top_n[self.training], num_anchors)
            _, top_n_idx = ob.topk(pre_nms_top_n, dim=1)
            r.append(top_n_idx + offset)
            offset += num_anchors
        return jt.concat(r, dim=1)

    def clip_boxes_to_image(self, boxes, img_shape):
        """将框裁剪到图像范围内"""
        h, w = img_shape
        boxes = boxes.clone()
        boxes[:, 0] = boxes[:, 0].clamp(min_v=0, max_v=w)
        boxes[:, 1] = boxes[:, 1].clamp(min_v=0, max_v=h)
        boxes[:, 2] = boxes[:, 2].clamp(min_v=0, max_v=w)
        boxes[:, 3] = boxes[:, 3].clamp(min_v=0, max_v=h)
        return boxes

    def execute(self, images, features, targets=None):
        """RPN的完整前向传播"""
        # 获取特征图上的预测
        objectness, pred_bbox_deltas = self.head(features)

        # 生成anchors
        anchors = self.anchor_generator(images, features)

        # 处理预测结果
        num_images = len(anchors)
        num_anchors_per_level = [
            o[0].shape[0] * o[0].shape[1] * o[0].shape[2] for o in objectness
        ]

        # 解码预测的边界框
        objectness, pred_bbox_deltas = concat_box_prediction_layers(
            objectness, pred_bbox_deltas
        )
        print(
            f"objectness: {objectness.shape}, pred_bbox_deltas: {pred_bbox_deltas.shape}"
        )
        proposals = self.box_coder.decode(pred_bbox_deltas.detach(), anchors[0])
        proposals = jt.view(proposals, (num_images, -1, 4))
        print(f"proposals: {proposals.shape}")
        boxes, scores = self.filter_proposals(
            proposals, objectness, images.image_sizes, num_anchors_per_level
        )
        print(f"boxes: {boxes[0].shape}, scores: {scores[0].shape}")

        losses = {}
        if self.training:
            assert targets is not None
            labels, matched_gt_boxes = self.assign_targets_to_anchors(anchors, targets)
            regression_targets = self.box_coder.encode(matched_gt_boxes, anchors)
            loss_objectness, loss_rpn_box_reg = self.compute_loss(
                objectness, pred_bbox_deltas, labels, regression_targets
            )
            losses = {
                "loss_objectness": loss_objectness,
                "loss_rpn_box_reg": loss_rpn_box_reg,
            }

        return boxes, losses

    def assign_targets_to_anchors(
        self, anchors: List[jt.Var], targets: Dict[str, jt.Var]
    ) -> Tuple[List[jt.Var], List[jt.Var]]:
        """为anchors分配ground truth标签

        Args:
            anchors: List[jt.Var] - 每张图像的anchors列表
            targets: Dict[str, jt.Var] - 训练目标
                - boxes: List[jt.Var] - GT boxes列表
                - labels: List[jt.Var] - GT labels列表

        Returns:
            labels: List[jt.Var] - 每个anchor的标签(1:正样本, 0:负样本, -1:忽略)
            matched_gt_boxes: List[jt.Var] - 每个anchor匹配到的GT box
        """
        labels = []
        matched_gt_boxes = []

        for anchors_per_image, targets_per_image in zip(
            anchors, zip(targets["boxes"], targets["labels"])
        ):
            gt_boxes = targets_per_image[0]

            if gt_boxes.numel() == 0:
                # 没有GT box时,所有anchor都是负样本
                device = anchors_per_image.device
                matched_gt_boxes_per_image = jt.zeros_like(anchors_per_image)
                labels_per_image = jt.zeros(
                    (anchors_per_image.shape[0],), dtype=jt.int32
                )
            else:
                # 计算anchors和GT boxes的IoU矩阵
                match_quality_matrix = box_ops.box_iou(gt_boxes, anchors_per_image)

                matched_vals, matches = match_quality_matrix.max(dim=0)

                labels_per_image = jt.full(
                    (anchors_per_image.shape[0],), -1, dtype=jt.int32
                )

                below_low_threshold = matched_vals < self.rpn_bg_iou_thresh
                labels_per_image[below_low_threshold] = 0

                above_high_threshold = matched_vals >= self.rpn_fg_iou_thresh
                labels_per_image[above_high_threshold] = 1

                highest_quality_foreach_gt, _ = match_quality_matrix.max(dim=1)
                _, highest_quality_indices = match_quality_matrix.max(dim=0)
                labels_per_image[highest_quality_indices] = 1

                # 获取匹配的GT boxes
                matched_gt_boxes_per_image = gt_boxes[matches]

            labels.append(labels_per_image)
            matched_gt_boxes.append(matched_gt_boxes_per_image)

        return labels, matched_gt_boxes

    def compute_loss(
        self,
        objectness: jt.Var,
        pred_bbox_deltas: jt.Var,
        labels: List[jt.Var],
        regression_targets: List[jt.Var],
    ) -> Tuple[jt.Var, jt.Var]:
        """计算RPN的损失

        Args:
            objectness: jt.Var - objectness预测 [N, -1]
            pred_bbox_deltas: jt.Var - 边界框回归预测 [N, -1, 4]
            labels: List[jt.Var] - anchor标签列表
            regression_targets: List[jt.Var] - 回归目标列表

        Returns:
            loss_objectness: jt.Var - objectness损失
            loss_rpn_box_reg: jt.Var - 边界框回归损失
        """
        sampled_pos_inds, sampled_neg_inds = self.subsample(labels)

        # 合并所有图像的采样索引
        sampled_pos_inds = jt.concat(sampled_pos_inds, dim=0)
        sampled_neg_inds = jt.concat(sampled_neg_inds, dim=0)
        sampled_inds = jt.concat([sampled_pos_inds, sampled_neg_inds], dim=0)

        # 合并所有图像的标签和回归目标
        labels = jt.concat(labels, dim=0)
        regression_targets = jt.concat(regression_targets, dim=0)

        # Objectness损失
        objectness = objectness.reshape(-1)

        labels = labels[sampled_inds]
        loss_objectness = jt.nn.binary_cross_entropy_with_logits(
            objectness[sampled_inds], labels.float()
        )

        # Box regression损失 - 只在正样本上计算
        if sampled_pos_inds.numel() > 0:
            pred_bbox_deltas = pred_bbox_deltas.reshape(-1, 4)

            loss_rpn_box_reg = (
                jt.nn.smooth_l1_loss(
                    pred_bbox_deltas[sampled_pos_inds],
                    regression_targets[sampled_pos_inds],
                    reduction="sum",
                )
                / sampled_inds.numel()
            )
        else:
            loss_rpn_box_reg = pred_bbox_deltas.sum() * 0

        return loss_objectness, loss_rpn_box_reg

    def subsample(self, labels: List[jt.Var]) -> Tuple[List[jt.Var], List[jt.Var]]:
        """对正负样本进行采样

        Args:
            labels: List[jt.Var] - anchor标签列表

        Returns:
            sampled_pos_inds: List[jt.Var] - 采样的正样本索引
            sampled_neg_inds: List[jt.Var] - 采样的负样本索引
        """
        sampled_pos_inds = []
        sampled_neg_inds = []

        for labels_per_image in labels:
            positive = jt.where(labels_per_image == 1)[0]
            negative = jt.where(labels_per_image == 0)[0]

            # 计算正样本数量
            num_pos = int(self.rpn_batch_size_per_image * self.rpn_positive_fraction)
            num_pos = min(positive.numel(), num_pos)

            # 计算负样本数量
            num_neg = self.rpn_batch_size_per_image - num_pos
            num_neg = min(negative.numel(), num_neg)

            # 随机采样
            perm1 = jt.randperm(positive.numel())[:num_pos]
            perm2 = jt.randperm(negative.numel())[:num_neg]

            pos_idx = positive[perm1]
            neg_idx = negative[perm2]

            sampled_pos_inds.append(pos_idx)
            sampled_neg_inds.append(neg_idx)

        return sampled_pos_inds, sampled_neg_inds
