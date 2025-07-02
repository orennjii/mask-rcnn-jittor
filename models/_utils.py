import math
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import jittor as jt
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torchvision.models.detection.image_list import ImageList
from torchvision.ops import (
    complete_box_iou_loss,
    distance_box_iou_loss,
    FrozenBatchNorm2d,
    generalized_box_iou_loss,
)
from torchvision.ops import boxes as box_ops


class AnchorGenerator(nn.Module):
    """
    Module that generates anchors for a set of feature maps and
    image sizes.

    The module support computing anchors at multiple sizes and aspect ratios
    per feature map. This module assumes aspect ratio = height / width for
    each anchor.

    sizes and aspect_ratios should have the same number of elements, and it should
    correspond to the number of feature maps.

    sizes[i] and aspect_ratios[i] can have an arbitrary number of elements,
    and AnchorGenerator will output a set of sizes[i] * aspect_ratios[i] anchors
    per spatial location for feature map i.

    Args:
        sizes (Tuple[Tuple[int]]):
        aspect_ratios (Tuple[Tuple[float]]):
    """

    __annotations__ = {
        "cell_anchors": List[torch.Tensor],
    }

    def __init__(
        self,
        sizes=((128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),),
    ):
        super().__init__()

        if not isinstance(sizes[0], (list, tuple)):
            # TODO change this
            sizes = tuple((s,) for s in sizes)
        if not isinstance(aspect_ratios[0], (list, tuple)):
            aspect_ratios = (aspect_ratios,) * len(sizes)

        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.cell_anchors = [
            self.generate_anchors(size, aspect_ratio)
            for size, aspect_ratio in zip(sizes, aspect_ratios)
        ]

    # TODO: https://github.com/pytorch/pytorch/issues/26792
    # For every (aspect_ratios, scales) combination, output a zero-centered anchor with those values.
    # (scales, aspect_ratios) are usually an element of zip(self.scales, self.aspect_ratios)
    # This method assumes aspect ratio = height / width for an anchor.
    def generate_anchors(
        self,
        scales: List[int],
        aspect_ratios: List[float],
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
    ) -> Tensor:
        scales = torch.as_tensor(scales, dtype=dtype, device=device)
        aspect_ratios = torch.as_tensor(aspect_ratios, dtype=dtype, device=device)
        h_ratios = torch.sqrt(aspect_ratios)
        w_ratios = 1 / h_ratios

        ws = (w_ratios[:, None] * scales[None, :]).view(-1)
        hs = (h_ratios[:, None] * scales[None, :]).view(-1)

        base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2
        return base_anchors.round()

    def set_cell_anchors(self, dtype: torch.dtype, device: torch.device):
        self.cell_anchors = [
            cell_anchor.to(dtype=dtype, device=device)
            for cell_anchor in self.cell_anchors
        ]

    def num_anchors_per_location(self) -> List[int]:
        return [len(s) * len(a) for s, a in zip(self.sizes, self.aspect_ratios)]

    # For every combination of (a, (g, s), i) in (self.cell_anchors, zip(grid_sizes, strides), 0:2),
    # output g[i] anchors that are s[i] distance apart in direction i, with the same dimensions as a.
    def grid_anchors(
        self, grid_sizes: List[List[int]], strides: List[List[Tensor]]
    ) -> List[Tensor]:
        anchors = []
        cell_anchors = self.cell_anchors
        torch._assert(cell_anchors is not None, "cell_anchors should not be None")
        torch._assert(
            len(grid_sizes) == len(strides) == len(cell_anchors),
            "Anchors should be Tuple[Tuple[int]] because each feature "
            "map could potentially have different sizes and aspect ratios. "
            "There needs to be a match between the number of "
            "feature maps passed and the number of sizes / aspect ratios specified.",
        )

        for size, stride, base_anchors in zip(grid_sizes, strides, cell_anchors):
            grid_height, grid_width = size
            stride_height, stride_width = stride
            device = base_anchors.device

            # For output anchor, compute [x_center, y_center, x_center, y_center]
            shifts_x = (
                torch.arange(0, grid_width, dtype=torch.int32, device=device)
                * stride_width
            )
            shifts_y = (
                torch.arange(0, grid_height, dtype=torch.int32, device=device)
                * stride_height
            )
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)

            # For every (base anchor, output anchor) pair,
            # offset each zero-centered base anchor by the center of the output anchor.
            anchors.append(
                (shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4)
            )

        return anchors

    def forward(
        self, image_list: ImageList, feature_maps: List[Tensor]
    ) -> List[Tensor]:
        grid_sizes = [feature_map.shape[-2:] for feature_map in feature_maps]
        image_size = image_list.tensors.shape[-2:]
        dtype, device = feature_maps[0].dtype, feature_maps[0].device
        strides = [
            [
                torch.empty((), dtype=torch.int64, device=device).fill_(
                    image_size[0] // g[0]
                ),
                torch.empty((), dtype=torch.int64, device=device).fill_(
                    image_size[1] // g[1]
                ),
            ]
            for g in grid_sizes
        ]
        self.set_cell_anchors(dtype, device)
        anchors_over_all_feature_maps = self.grid_anchors(grid_sizes, strides)
        anchors: List[List[torch.Tensor]] = []
        for _ in range(len(image_list.image_sizes)):
            anchors_in_image = [
                anchors_per_feature_map
                for anchors_per_feature_map in anchors_over_all_feature_maps
            ]
            anchors.append(anchors_in_image)
        anchors = [torch.cat(anchors_per_image) for anchors_per_image in anchors]
        return anchors


class BoxCoder:
    """
    This class encodes and decodes a set of bounding boxes into
    the representation used for training the regressors.
    """

    def __init__(
        self,
        weights: Tuple[float, float, float, float],
        bbox_xform_clip: float = math.log(1000.0 / 16),
    ) -> None:
        """
        Args:
            weights (4-element tuple)
            bbox_xform_clip (float)
        """
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip

    def encode(
        self, reference_boxes: List[Tensor], proposals: List[Tensor]
    ) -> List[Tensor]:
        boxes_per_image = [len(b) for b in reference_boxes]
        reference_boxes = torch.cat(reference_boxes, dim=0)
        proposals = torch.cat(proposals, dim=0)
        targets = self.encode_single(reference_boxes, proposals)
        return targets.split(boxes_per_image, 0)

    def encode_single(self, reference_boxes: Tensor, proposals: Tensor) -> Tensor:
        """
        Encode a set of proposals with respect to some
        reference boxes

        Args:
            reference_boxes (Tensor): reference boxes
            proposals (Tensor): boxes to be encoded
        """
        dtype = reference_boxes.dtype
        device = reference_boxes.device
        weights = torch.as_tensor(self.weights, dtype=dtype, device=device)
        targets = encode_boxes(reference_boxes, proposals, weights)

        return targets

    def decode(self, rel_codes: Tensor, boxes: List[Tensor]) -> Tensor:
        torch._assert(
            isinstance(boxes, (list, tuple)),
            "This function expects boxes of type list or tuple.",
        )
        torch._assert(
            isinstance(rel_codes, torch.Tensor),
            "This function expects rel_codes of type torch.Tensor.",
        )
        boxes_per_image = [b.size(0) for b in boxes]
        concat_boxes = torch.cat(boxes, dim=0)
        box_sum = 0
        for val in boxes_per_image:
            box_sum += val
        if box_sum > 0:
            rel_codes = rel_codes.reshape(box_sum, -1)
        pred_boxes = self.decode_single(rel_codes, concat_boxes)
        if box_sum > 0:
            pred_boxes = pred_boxes.reshape(box_sum, -1, 4)
        return pred_boxes

    def decode_single(self, rel_codes: Tensor, boxes: Tensor) -> Tensor:
        """
        From a set of original boxes and encoded relative box offsets,
        get the decoded boxes.

        Args:
            rel_codes (Tensor): encoded boxes
            boxes (Tensor): reference boxes.
        """

        boxes = boxes.to(rel_codes.dtype)

        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        wx, wy, ww, wh = self.weights
        dx = rel_codes[:, 0::4] / wx
        dy = rel_codes[:, 1::4] / wy
        dw = rel_codes[:, 2::4] / ww
        dh = rel_codes[:, 3::4] / wh

        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=self.bbox_xform_clip)
        dh = torch.clamp(dh, max=self.bbox_xform_clip)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        # Distance from center to box's corner.
        c_to_c_h = (
            torch.tensor(0.5, dtype=pred_ctr_y.dtype, device=pred_h.device) * pred_h
        )
        c_to_c_w = (
            torch.tensor(0.5, dtype=pred_ctr_x.dtype, device=pred_w.device) * pred_w
        )

        pred_boxes1 = pred_ctr_x - c_to_c_w
        pred_boxes2 = pred_ctr_y - c_to_c_h
        pred_boxes3 = pred_ctr_x + c_to_c_w
        pred_boxes4 = pred_ctr_y + c_to_c_h
        pred_boxes = torch.stack(
            (pred_boxes1, pred_boxes2, pred_boxes3, pred_boxes4), dim=2
        ).flatten(1)
        return pred_boxes


def encode_boxes(reference_boxes: Tensor, proposals: Tensor, weights: Tensor) -> Tensor:
    """
    Encode a set of proposals with respect to some
    reference boxes

    Args:
        reference_boxes (Tensor): reference boxes
        proposals (Tensor): boxes to be encoded
        weights (Tensor[4]): the weights for ``(x, y, w, h)``
    """

    # perform some unpacking to make it JIT-fusion friendly
    wx = weights[0]
    wy = weights[1]
    ww = weights[2]
    wh = weights[3]

    proposals_x1 = proposals[:, 0].unsqueeze(1)
    proposals_y1 = proposals[:, 1].unsqueeze(1)
    proposals_x2 = proposals[:, 2].unsqueeze(1)
    proposals_y2 = proposals[:, 3].unsqueeze(1)

    reference_boxes_x1 = reference_boxes[:, 0].unsqueeze(1)
    reference_boxes_y1 = reference_boxes[:, 1].unsqueeze(1)
    reference_boxes_x2 = reference_boxes[:, 2].unsqueeze(1)
    reference_boxes_y2 = reference_boxes[:, 3].unsqueeze(1)

    # implementation starts here
    ex_widths = proposals_x2 - proposals_x1
    ex_heights = proposals_y2 - proposals_y1
    ex_ctr_x = proposals_x1 + 0.5 * ex_widths
    ex_ctr_y = proposals_y1 + 0.5 * ex_heights

    gt_widths = reference_boxes_x2 - reference_boxes_x1
    gt_heights = reference_boxes_y2 - reference_boxes_y1
    gt_ctr_x = reference_boxes_x1 + 0.5 * gt_widths
    gt_ctr_y = reference_boxes_y1 + 0.5 * gt_heights

    targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = ww * torch.log(gt_widths / ex_widths)
    targets_dh = wh * torch.log(gt_heights / ex_heights)

    targets = torch.cat((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)
    return targets


class JittorBoxCoder:
    def __init__(
        self,
        weights=(1.0, 1.0, 1.0, 1.0),
        bbox_xform_clip: float = math.log(1000.0 / 16),
    ):
        self.weights = weights
        # 创建torch的BoxCoder实例
        self.torch_box_coder = BoxCoder(
            weights=weights, bbox_xform_clip=bbox_xform_clip
        )

    def encode(self, reference_boxes, proposals):
        # 将jittor变量转换为torch tensor
        reference_boxes_torch = torch.from_numpy(reference_boxes.numpy())
        proposals_torch = torch.from_numpy(proposals.numpy())

        # 使用torch BoxCoder进行编码
        targets_torch = self.torch_box_coder.encode_single(
            reference_boxes_torch, proposals_torch
        )

        # 将结果转回jittor格式
        targets = jt.array(targets_torch.numpy())
        return targets

    def decode(self, rel_codes: jt.Var, boxes: jt.Var):
        """将预测的偏移量解码为实际的边界框坐标
        Args:
            rel_codes: jt.Var [N, num_classes * 4] 预测的边界框偏移量
            boxes: jt.Var [N, 4] anchor boxes
        Returns:
            jt.Var [N, num_classes, 4] 解码后的边界框坐标
        """
        # 将jittor变量转换为torch tensor
        rel_codes_torch = torch.from_numpy(rel_codes.numpy())
        boxes_torch = torch.from_numpy(boxes.numpy())

        # 将boxes包装成列表，因为torch BoxCoder.decode需要列表输入
        boxes_list = [boxes_torch]

        # 使用torch BoxCoder进行解码
        pred_boxes_torch = self.torch_box_coder.decode(rel_codes_torch, boxes_list)

        # 将结果转回jittor格式
        pred_boxes = jt.array(pred_boxes_torch.numpy())
        print(f"pred_boxes shape: {pred_boxes.shape}")

        return pred_boxes


def remove_small_boxes(boxes, min_size):
    """移除小框"""
    ws = boxes[:, 2] - boxes[:, 0]
    hs = boxes[:, 3] - boxes[:, 1]
    keep = (ws >= min_size) & (hs >= min_size)
    return keep


def torch_batched_nms(boxes, scores, idxs, iou_threshold):
    """
    使用 torchvision 的 batched_nms 实现多批次 NMS

    Args:
        boxes (jt.Var): shape (N, 4), 边界框坐标
        scores (jt.Var): shape (N,), 置信度分数
        idxs (jt.Var): shape (N,), 类别索引
        iou_threshold (float): NMS 的 IoU 阈值

    Returns:
        jt.Var: 保留框的索引
    """
    # 检查输入是否为空
    if boxes.numel() == 0:
        return jt.zeros(0, dtype="int64")

    # 转换为 PyTorch tensor
    boxes_torch = torch.from_numpy(boxes.numpy())
    scores_torch = torch.from_numpy(scores.numpy())
    idxs_torch = torch.from_numpy(idxs.numpy())

    # 确保数据类型正确
    boxes_torch = boxes_torch.float()
    scores_torch = scores_torch.float()
    idxs_torch = idxs_torch.long()

    # 使用 torchvision 的 batched_nms
    keep_torch = box_ops.batched_nms(
        boxes=boxes_torch,
        scores=scores_torch,
        idxs=idxs_torch,
        iou_threshold=iou_threshold,
    )

    # 将结果转回 Jittor 变量
    keep = jt.array(keep_torch.numpy())
    return keep


def jittor_batched_nms(boxes, scores, idxs, iou_threshold):
    """
    对不同层级的框分别进行NMS

    Args:
        boxes (jt.Var): shape (N, 4), 边界框坐标
        scores (jt.Var): shape (N,), 置信度分数
        idxs (jt.Var): shape (N,), 类别索引
        iou_threshold (float): NMS 的 IoU 阈值
    """
    if boxes.numel() == 0:
        return jt.zeros(0, dtype="int64")

    # 为每个层级的框添加偏移，确保不同层级的框不会重叠
    max_coordinate = boxes.max()
    offsets = idxs.to(boxes.dtype) * (max_coordinate + 1)
    boxes_for_nms = boxes.clone()
    boxes_for_nms[:, :2] += offsets[:, None]
    boxes_for_nms[:, 2:] += offsets[:, None]
    boxes_and_scores = jt.concat([boxes_for_nms, scores[:, None]], dim=1)

    # 执行NMS
    keep = jt.nms(boxes_and_scores, iou_threshold)
    return keep
