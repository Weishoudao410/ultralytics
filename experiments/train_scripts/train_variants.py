from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import torch

# ========== 路径设置 ==========
ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "experiments" / "models"

# 确保本地仓库优先导入
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 避免 IDE/解释器预加载 site-packages 的 ultralytics
for _m in [m for m in list(sys.modules) if m == "ultralytics" or m.startswith("ultralytics.")]:
    sys.modules.pop(_m, None)

from ultralytics import YOLO  # noqa: E402
import ultralytics  # noqa: E402

# ========== 环境变量读取 ==========
DATA = os.getenv("DATA", "dataset.yaml")
EPOCHS = int(os.getenv("EPOCHS", "100"))
IMGSZ = int(os.getenv("IMGSZ", "640"))
BATCH = int(os.getenv("BATCH", "16"))
DEVICE = os.getenv("DEVICE", "0")
PROJECT = os.getenv("PROJECT", "runs/improve_train")


def _env(name: str, default: str) -> str:
    """获取环境变量，如果不存在则使用默认值"""
    value = os.getenv(name)
    return default if value is None else value


def _register_custom_modules_for_runtime() -> None:
    """将实验模块注入当前运行时 parser 命名空间，避免 YAML 解析 KeyError。"""
    from ultralytics.nn import tasks as nn_tasks
    from ultralytics.nn.modules import BiFANFusion, BiFormerBlock, CoordAtt, DWConvBlock

    nn_tasks.DWConvBlock = DWConvBlock
    nn_tasks.CoordAtt = CoordAtt
    nn_tasks.BiFormerBlock = BiFormerBlock
    nn_tasks.BiFANFusion = BiFANFusion


def _resolve_device(device: str) -> str:
    """当无 CUDA 时自动回退到 CPU，避免 device=0 报错。"""
    if device.lower() == "cpu":
        return "cpu"
    if "cuda" in device.lower() and not torch.cuda.is_available():
        warnings.warn("CUDA 不可用，自动切换到 device=cpu。", stacklevel=1)
        return "cpu"
    if any(ch.isdigit() for ch in device) and not torch.cuda.is_available():
        warnings.warn("检测到 device 使用 GPU 索引，但当前无可用 CUDA，自动切换到 device=cpu。", stacklevel=1)
        return "cpu"
    return device


def train_all_variants() -> None:
    """训练所有模型变体"""
    local_ultralytics = Path(ultralytics.__file__).resolve()
    if ROOT not in local_ultralytics.parents:
        warnings.warn(
            "Detected site-packages ultralytics import. Attempting to continue, but custom modules "
            "(DWConvBlock/CoordAtt/BiFormerBlock/BiFANFusion) may be missing. "
            "If parsing fails, run: pip uninstall -y ultralytics && pip install -e .",
            stacklevel=1,
        )
        raise RuntimeError(
            "Detected site-packages ultralytics import. Please run from the local repo with editable install: "
            "pip uninstall -y ultralytics && pip install -e ."
        )

    _register_custom_modules_for_runtime()

    variants = [
        (MODEL_DIR / "yolov8s_dwconv_backbone.yaml", "v1_dwconv_backbone", False),
        (MODEL_DIR / "yolov8s_ca_before_c2f.yaml", "v2_ca_before_c2f", False),
        (MODEL_DIR / "yolov8s_biformer_after_sppf.yaml", "v3_biformer_after_sppf", False),
        (MODEL_DIR / "yolov8s_bifan_neck.yaml", "v4_bifan_neck", False),
        (MODEL_DIR / "yolov8s_siou_loss.yaml", "v5_siou_loss", True),
    ]

    for model_yaml, run_name, use_siou in variants:
        if not model_yaml.exists():
            raise FileNotFoundError(f"Model YAML not found: {model_yaml}")

        # 设置是否使用 SiOU loss
        os.environ["YOLO_USE_SIOU"] = "true" if use_siou else "false"

        run_device = _resolve_device(_env("DEVICE", DEVICE))

        # 打印提示当前训练的变体
        print(f"\n=== 开始训练模型变体: {run_name} ===")
        print(f"模型 YAML: {model_yaml}")
        print(f"使用 SiOU Loss: {use_siou}")
        print(f"数据集: {_env('DATA', DATA)}, Epochs: {EPOCHS}, ImgSz: {IMGSZ}, Batch: {BATCH}, Device: {run_device}\n")

        # 初始化模型并加载预训练权重
        model = YOLO(str(model_yaml)).load("yolov8s.pt")

        # 开始训练
        model.train(
            data=_env("DATA", DATA),
            epochs=int(_env("EPOCHS", str(EPOCHS))),
            imgsz=int(_env("IMGSZ", str(IMGSZ))),
            batch=int(_env("BATCH", str(BATCH))),
            device=run_device,
            project=_env("PROJECT", PROJECT),
            name=run_name,
        )


if __name__ == "__main__":
    train_all_variants()
