import cv2
import numpy as np
import matplotlib.pyplot as plt

# COCO数据集的类别名称（81个类别，包括背景类）
COCO_CLASSES = [
    "background",  # 索引0为背景
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


# 添加一个映射函数来处理COCO的类别索引
def coco_category_id_to_label(category_id):
    """
    将COCO的类别ID转换为连续的标签索引
    Args:
        category_id: COCO数据集中的类别ID（1-90）
    Returns:
        label: 连续的标签索引（0-80，其中0为背景）
    """
    # COCO类别ID到连续标签的映射
    if category_id == 0:  # 背景类
        return 0

    # COCO数据集中实际使用的类别ID列表
    coco_ids = [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        27,
        28,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        60,
        61,
        62,
        63,
        64,
        65,
        67,
        70,
        72,
        73,
        74,
        75,
        76,
        77,
        78,
        79,
        80,
        81,
        82,
        84,
        85,
        86,
        87,
        88,
        89,
        90,
    ]

    # 找到category_id在coco_ids中的位置，加1（因为0是背景）
    try:
        return coco_ids.index(category_id) + 1
    except ValueError:
        return 0  # 如果找不到对应的类别ID，返回背景类


def visualize_prediction(img, masks, boxes, labels, scores):
    # 创建一个新的图像用于显示
    result_img = img.copy()

    # 为不同的实例随机生成不同的颜色
    colors = np.random.randint(0, 255, size=(len(masks), 3))

    # 获取原始图像尺寸
    h, w = img.shape[:2]

    # 绘制每个检测到的实例
    for i in range(len(masks)):
        if scores[i] >= 0.5:
            # masks已经是numpy数组，直接使用
            mask = masks[i][0]  # 取第一个通道
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR) > 0.5
            color = colors[i]

            # 应用mask
            # colored_mask = np.zeros_like(img)
            # colored_mask[mask] = color
            # result_img = cv2.addWeighted(result_img, 1, colored_mask, 0.5, 0)

            # 直接将mask区域替换为纯色
            result_img[mask] = color

            # 绘制边界框
            box = boxes[i].astype(np.int32)  # 直接使用numpy数组
            cv2.rectangle(
                result_img, (box[0], box[1]), (box[2], box[3]), color.tolist(), 2
            )

            # 添加类别标签和置信度
            label = COCO_CLASSES[coco_category_id_to_label(int(labels[i]))]
            score = scores[i]
            text = f"{label}: {score:.2f}"
            cv2.putText(
                result_img,
                text,
                (box[0], box[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color.tolist(),
                2,
            )

    return result_img


def save_visualization(img, result_img, output_path):
    plt.figure(figsize=(15, 10))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(img)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Detection Result")
    plt.imshow(result_img)
    plt.axis("off")

    # 保存结果
    plt.savefig(output_path)
    print(f"结果已保存到: {output_path}")

    # 显示结果
    plt.show()


def print_detection_results(masks, labels, scores):
    print(f"检测到 {len(masks)} 个对象")
    for i in range(len(masks)):
        print(
            f"对象 {i+1}: {COCO_CLASSES[coco_category_id_to_label(int(labels[i]))]} (置信度: {scores[i]:.2f})"
        )
