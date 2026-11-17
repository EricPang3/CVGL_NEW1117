"""AutoDL base 环境检查：不下载模型、不读取数据、不修改环境。"""

import platform
import sys
from importlib.metadata import version


def main():
    print(f"Python: {platform.python_version()}；系统: {platform.platform()}")
    try:
        import torch
        import torchvision
        import timm
        import albumentations
        import cv2
        import numpy
        import yaml
        from torchvision.ops import nms
        for package in ("torch", "torchvision", "timm", "numpy", "albumentations", "opencv-python-headless", "PyYAML", "Pillow", "huggingface-hub"):
            print(f"{package}: {version(package)}")
    except (ImportError, RuntimeError) as error:
        print(f"依赖导入失败：{error}\n请按使用教程安装依赖或修复 torchvision 匹配问题。", file=sys.stderr)
        return 1
    errors = []
    if sys.version_info[:2] != (3, 10):
        errors.append("目标环境需要 Python 3.10；请使用 AutoDL base")
    if torch.__version__.split("+")[0] != "2.1.2" or torchvision.__version__.split("+")[0] != "0.16.2":
        errors.append("目标框架版本为 torch 2.1.2 / torchvision 0.16.2")
    print(f"PyTorch CUDA runtime: {torch.version.cuda}；CUDA 可用: {torch.cuda.is_available()}")
    if torch.version.cuda != "11.8" or not torch.cuda.is_available():
        errors.append("需要 CUDA 11.8 的 GPU 版 PyTorch，以及可用的 GPU")
    else:
        count = torch.cuda.device_count()
        name = torch.cuda.get_device_name(0)
        print(f"可见 GPU: {count}；设备 0: {name}")
        if count != 1 or "4090" not in name:
            errors.append("目标配置为单张 RTX 4090；多卡机器可用 CUDA_VISIBLE_DEVICES 选择一张卡")
        boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0]], device="cuda:0")
        # 实际执行 torchvision CUDA 算子，检查 torch/torchvision 二进制匹配。
        nms(boxes, torch.ones(1, device="cuda:0"), 0.5)
        torch.cuda.synchronize()
    for error in errors:
        print(f"未通过：{error}", file=sys.stderr)
    if not errors:
        print("AutoDL 环境检查通过。")
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
