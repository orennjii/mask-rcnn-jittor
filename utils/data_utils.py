import jittor as jt
from jittor import nn


class ImageList:
    """
    用于jittor模型的图像列表数据结构，存储批量图像及其原始尺寸

    Args:
        tensors (jt.Var): 批量图像张量 [N, C, H, W]
        image_sizes (List[Tuple[int, int]]): 原始图像尺寸列表 [(h1, w1), (h2, w2), ...]
    """

    def __init__(self, tensors, image_sizes=None):
        self.tensors = tensors
        self.image_sizes = (
            (tensors.shape[-2], tensors.shape[-1])
            if image_sizes is None
            else image_sizes
        )

    def to(self):
        cast_tensor = self.tensors
        return ImageList(cast_tensor, self.image_sizes)


def jittor_resize_image(image, min_size=800, max_size=1333):
    """
    用于Jittor模型的图像预处理

    Args:
        image: numpy数组，(H, W, C)
        min_size: 最小尺寸
        max_size: 最大尺寸
    """
    origin_height, origin_width = image.shape[-2:]

    # 计算缩放比例
    scale = min_size / min(origin_height, origin_width)

    # 保持长宽比
    if origin_height < origin_width:
        new_height = min_size
        new_width = int(min(origin_width * scale, max_size))
    else:
        new_width = min_size
        new_height = int(min(origin_height * scale, max_size))

    # 缩放图像
    image = nn.interpolate(image, size=(new_height, new_width), mode="bilinear")

    pad_height = (32 - new_height % 32) % 32
    pad_width = (32 - new_width % 32) % 32

    image = nn.pad(image, [0, pad_width, 0, pad_height], mode="constant", value=0)

    image_list = ImageList(image, [(new_height, new_width)])

    return image_list
