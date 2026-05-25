from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import torch

# ===================== 路径设置 =====================
ROOT = Path(__file__).resolve().parents[2]  # 仓库根目录
MODEL_DIR = ROOT / "experiments" / "models"

# 确保本地仓库优先导入
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ===================== 导入 YOLO =====================
import ultralytics  # noqa: E402
from ultralytics import YOLO  # noqa: E402

# ===================== 环境变量 & 默认参数 =====================
DATA = os.getenv("DATA", str(ROOT / "dataset.yaml"))  # 数据集路径
EPOCHS = int(os.getenv("EPOCHS", "100"))
IMGSZ = int(os.getenv("IMGSZ", "640"))
BATCH = int(os.getenv("BATCH", "16"))
DEVICE = os.getenv("DEVICE", "cpu")  # 默认使用 CPU
PROJECT = os.getenv("PROJECT", str(ROOT / "runs/improve_train"))


# ===================== 自定义模块注册 =====================
def register_custom_modules() -> None:
    from ultralytics.nn import tasks as nn_tasks
    from ultralytics.nn.modules import BiFANFusion, BiFormerBlock, CoordAtt, DWConvBlock

    nn_tasks.DWConvBlock = DWConvBlock
    nn_tasks.CoordAtt = CoordAtt
    nn_tasks.BiFormerBlock = BiFormerBlock
    nn_tasks.BiFANFusion = BiFANFusion


# ===================== 设备解析 =====================
def resolve_device(device: str) -> str:
    """自动回退到 CPU"""
    if device.lower() == "cpu":
        return "cpu"
    if "cuda" in device.lower() and not torch.cuda.is_available():
        warnings.warn("CUDA 不可用，自动切换到 CPU。", stacklevel=1)
        return "cpu"
    if device.isdigit() and not torch.cuda.is_available():
        warnings.warn("检测到 GPU 索引，但无可用 CUDA，自动切换到 CPU。", stacklevel=1)
        return "cpu"
    return device


def resolve_data_path(data: str) -> str:
    """Resolve dataset yaml path; support common project layouts."""
    p = Path(data)
    if p.exists():
        return str(p)
    # common case: dataset.yaml is placed one level above repo clone folder
    candidate = ROOT.parent / "dataset.yaml"
    if candidate.exists():
        warnings.warn(f"默认数据集文件不存在，自动使用: {candidate}", stacklevel=1)
        return str(candidate)
    raise FileNotFoundError(
        f"Dataset yaml not found: {p}. Please set DATA env, e.g. DATA=C:/path/to/dataset.yaml"
    )


# ===================== 模型变体列表 =====================
VARIANTS = [
    ("yolov8s_dwconv_backbone.yaml", "v1_dwconv_backbone", False),
    ("yolov8s_ca_before_c2f.yaml", "v2_ca_before_c2f", False),
    ("yolov8s_biformer_after_sppf.yaml", "v3_biformer_after_sppf", False),
    ("yolov8s_bifan_neck.yaml", "v4_bifan_neck", False),
    ("yolov8s_siou_loss.yaml", "v5_siou_loss", True),
]


# ===================== 训练函数 =====================
def train_all_variants() -> None:
    # 检测是否使用本地 ultralytics
    if "site-packages" in str(Path(ultralytics.__file__).resolve()):
        warnings.warn(
            "Detected site-packages ultralytics import. Custom modules may be missing.",
            stacklevel=1,
        )

    register_custom_modules()
    run_device = resolve_device(DEVICE)
    data_path = resolve_data_path(DATA)

    for yaml_name, run_name, use_siou in VARIANTS:
        model_yaml = MODEL_DIR / yaml_name
        if not model_yaml.exists():
            raise FileNotFoundError(f"Model YAML not found: {model_yaml}")

        # 设置 SiOU Loss 环境变量
        os.environ["YOLO_USE_SIOU"] = "true" if use_siou else "false"

        print(f"\n=== 开始训练模型变体: {run_name} ===")
        print(f"模型 YAML: {model_yaml}")
        print(f"使用 SiOU Loss: {use_siou}")
        print(f"数据集: {data_path}, Epochs: {EPOCHS}, ImgSz: {IMGSZ}, Batch: {BATCH}, Device: {run_device}\n")

        # 初始化模型
        model = YOLO(str(model_yaml))

        # 尝试加载预训练权重
        pt_path = ROOT / "yolov8s.pt"
        if pt_path.exists():
            model = model.load(str(pt_path))
        else:
            warnings.warn(f"未找到本地预训练权重: {pt_path}，将使用默认下载或随机初始化。", stacklevel=1)

        # 开始训练
        model.train(
            data=data_path,
            epochs=EPOCHS,
            imgsz=IMGSZ,
            batch=BATCH,
            device=run_device,
            project=str(PROJECT),
            name=run_name,
        )


# ===================== 主程序 =====================
if __name__ == "__main__":
    train_all_variants()
