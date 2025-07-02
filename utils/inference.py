import torch
import jittor as jt
from jittor import nn
from jittor.transform import Resize
import cv2
import numpy as np
from pathlib import Path
from .visualization import COCO_CLASSES
from .data_utils import ImageList


def get_prediction(model, image_path, threshold=0.5, device="cpu"):
    """
    获取模型预测结果
    Args:
        model: PyTorch或Jittor模型
        image_path: 图像路径
        threshold: 置信度阈值
        device: 设备（仅PyTorch模型需要）
    """
    # 读取图像
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")
    origin_size = img.shape[:2]
    print(f"origin_size: {origin_size}")

    # 图像预处理
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 判断模型类型并进行相应处理
    is_torch_model = isinstance(model, torch.nn.Module)

    if is_torch_model:
        # PyTorch模型处理
        img_tensor = torch.from_numpy(img / 255.0).permute(2, 0, 1).float()
        img_tensor = img_tensor.to(device)
        if device:
            model = model.to(device)
            img_tensor = img_tensor.to(device)
        model.eval()
        with torch.no_grad():
            prediction = model([img_tensor])
            prediction = prediction[0]  # 取第一个batch的结果
    else:
        # Jittor模型处理
        img_var = jt.array(np.expand_dims(img, axis=0))
        img_var = jt.transpose(img_var, (0, 3, 1, 2)) / 255.0
        img_var = img_var.float32()
        print(f"img_var: {img_var.shape}")

        model.eval()
        with jt.no_grad():
            prediction = model(img_var)

    # 获取预测结果
    if is_torch_model:
        # PyTorch结果处理
        scores = prediction["scores"]
        high_scores_idxs = scores > threshold
        masks = prediction["masks"][high_scores_idxs]
        boxes = prediction["boxes"][high_scores_idxs]
        labels = prediction["labels"][high_scores_idxs]
        scores = scores[high_scores_idxs]

        # 转换为numpy数组
        masks = masks.cpu().numpy()
        boxes = boxes.cpu().numpy()
        labels = labels.cpu().numpy()
        scores = scores.cpu().numpy()
        print(
            f"masks: {masks.shape},\nboxes: {boxes.shape},\nlabels: {labels.shape},\nscores: {scores.shape}\n"
        )
    else:
        # Jittor结果处理
        scores = prediction["scores"]
        high_scores_idxs = scores > threshold
        masks = prediction["masks"][high_scores_idxs]
        boxes = prediction["boxes"][high_scores_idxs]
        labels = prediction["labels"][high_scores_idxs]
        scores = scores[high_scores_idxs]

        # 转换为numpy数组
        masks = masks.numpy()
        boxes = boxes.numpy()
        labels = labels.numpy()
        scores = scores.numpy()
        print(
            f"masks: {masks.shape},\nboxes: {boxes.shape},\nlabels: {labels.shape},\nscores: {scores.shape}\n"
        )

    return img, masks, boxes, labels, scores


def print_detection_results(masks, labels, scores):
    print(f"检测到 {len(masks)} 个对象")
    for i in range(len(masks)):
        print(f"对象 {i+1}: {COCO_CLASSES[int(labels[i])]} (置信度: {scores[i]:.2f})")
