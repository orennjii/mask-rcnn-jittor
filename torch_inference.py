import torch
from models.mask_rcnn import MaskRCNN
from pathlib import Path
from utils import (
    get_prediction,
    print_detection_results,
    visualize_prediction,
    save_visualization,
)
import configparser

ROOT_DIR = Path(__file__).parent.absolute()
config = configparser.ConfigParser()
config.read(ROOT_DIR / "config.cfg")


def register_shape_hooks(model):
    """
    为模型的每一层注册hook来打印输出的shape
    """
    hooks = []

    def get_shape_info(obj):
        """递归地获取复杂数据结构中的形状信息"""
        if isinstance(obj, (list, tuple)):
            return [get_shape_info(o) for o in obj]
        elif isinstance(obj, dict):
            return {k: get_shape_info(v) for k, v in obj.items()}
        elif isinstance(obj, torch.Tensor):
            return obj.shape
        else:
            return type(obj)

    def hook_fn(module, input, output):
        module_name = module.__class__.__name__
        shape_info = get_shape_info(output)
        print(f"\n{module_name} output shapes:", shape_info)

    def register_hook_recursive(module):
        hooks.append(module.register_forward_hook(hook_fn))
        for child in module.children():
            register_hook_recursive(child)

    register_hook_recursive(model)
    return hooks


def main():
    # 初始化模型
    model = MaskRCNN(
        num_classes=int(config["MODEL"]["num_classes"]),
        weights_path=str(ROOT_DIR / config["MODEL"]["weights_path"]),
    )

    # 注册hooks
    hooks = register_shape_hooks(model)

    image_path = "D:/code/dataset/coco_2017/train2017/000000006896.jpg"
    img, masks, boxes, labels, scores = get_prediction(
        model, image_path, threshold=0.8, device=str(config["DEVICE"]["device"])
    )

    # 移除hooks
    for hook in hooks:
        hook.remove()

    if len(masks) == 0:
        print("没有检测到任何对象！")
        return
    else:
        print_detection_results(masks, labels, scores)

    result_img = visualize_prediction(img, masks, boxes, labels, scores)

    output_path = ROOT_DIR / "testdata/result_torch.png"
    save_visualization(img, result_img, output_path)


if __name__ == "__main__":
    main()
