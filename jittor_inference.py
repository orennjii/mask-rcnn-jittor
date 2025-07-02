from jittor_inference_model import JittorMaskRCNN
import jittor as jt
from utils import (
    get_prediction,
    print_detection_results,
    visualize_prediction,
    save_visualization,
)
import configparser
import numpy as np
import cv2
from pathlib import Path

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
        elif isinstance(obj, jt.Var):
            return obj.shape
        else:
            return type(obj)

    # 修改hook函数签名，适配Jittor的要求
    def hook_fn(module, *args):
        module_name = module.__class__.__name__
        # Jittor的hook函数中，最后一个参数是output
        output = args[-1]
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
    model = JittorMaskRCNN(weights_path=str(ROOT_DIR / config["MODEL"]["weights_path"]))

    # hooks = register_shape_hooks(model)

    image_path = "D:/code/dataset/coco_2017/train2017/000000006896.jpg"  # 000000005344.jpg, 000000001270.jpg
    # image_path = str(ROOT_DIR / "testdata/test.jpg")
    img, masks, boxes, labels, scores = get_prediction(
        model,
        image_path,
        threshold=0.5,
        device=str(ROOT_DIR / config["DEVICE"]["device"]),
    )

    if len(masks) == 0:
        print("没有检测到任何对象！")
        return
    else:
        print_detection_results(masks, labels, scores)

    result_img = visualize_prediction(img, masks, boxes, labels, scores)
    cv2.imwrite(str(ROOT_DIR / "testdata/result_jittor.jpg"), result_img)

    output_path = ROOT_DIR / "testdata/contrast_result_jittor.jpg"
    save_visualization(img, result_img, output_path)


if __name__ == "__main__":
    main()
