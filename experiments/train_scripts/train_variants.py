from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import torch

# ===================== 路径设置 =====================
ROOT = Path(__file__).resolve().parents[2]  # 仓库根目录
# ========== 路径设置 ==========
ROOT = Path(__file__).resolve().parents[2]
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
# 避免 IDE/解释器预加载 site-packages 的 ultralytics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "experiments" / "models"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Force local repo import precedence even if site-packages ultralytics was preloaded by IDE/runtime.
for _m in [m for m in list(sys.modules) if m == "ultralytics" or m.startswith("ultralytics.")]:
    sys.modules.pop(_m, None)

from ultralytics import YOLO  # noqa: E402
import ultralytics  # noqa: E402

# ========== 环境变量读取 ==========
DATA = os.getenv("DATA", r"C:\Users\13681\PycharmProjects\v8\dataset.yaml")
DATA = os.getenv("DATA", "dataset.yaml")
from ultralytics import YOLO  # noqa: E402


def _register_custom_modules_for_runtime() -> None:
    """Register local experimental modules into whichever ultralytics package is imported at runtime."""
    from ultralytics.nn import tasks as nn_tasks
    from ultralytics.nn.modules import BiFANFusion, BiFormerBlock, CoordAtt, DWConvBlock

    # Make parse_model globals()[m] resolvable even when ultralytics comes from site-packages.
    nn_tasks.DWConvBlock = DWConvBlock
    nn_tasks.CoordAtt = CoordAtt
    nn_tasks.BiFormerBlock = BiFormerBlock
    nn_tasks.BiFANFusion = BiFANFusion
import warnings
from pathlib import Path

# Resolve repository root from this script path.
ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "experiments" / "models"

# Ensure local repository is preferred in import resolution.
from pathlib import Path

from pathlib import Path

from ultralytics import YOLO

DATA = os.getenv("DATA", "coco8.yaml")
EPOCHS = int(os.getenv("EPOCHS", "100"))
IMGSZ = int(os.getenv("IMGSZ", "640"))
BATCH = int(os.getenv("BATCH", "16"))
DEVICE = os.getenv("DEVICE", "0")
PROJECT = os.getenv("PROJECT", "runs/improve_train")


def _env(name: str, default: str) -> str:
    """获取环境变量，如果不存在则使用默认值"""
ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "experiments" / "models"

# Ensure the local repository package is imported instead of site-packages.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO  # noqa: E402


def _env(name: str, default: str) -> str:
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
def train_all_variants() -> None:
    local_ultralytics = Path(ultralytics.__file__).resolve()
    if ROOT not in local_ultralytics.parents:
        raise RuntimeError(
            f"Expected local ultralytics import under {ROOT}, but got {local_ultralytics}. "
            "Please run in repo environment or reinstall editable: pip uninstall -y ultralytics && pip install -e ."
        )

    _register_custom_modules_for_runtime()

    data = _env("DATA", "coco8.yaml")
    epochs = int(_env("EPOCHS", "100"))
    imgsz = int(_env("IMGSZ", "640"))
    batch = int(_env("BATCH", "16"))
    device = _env("DEVICE", "0")
    project = _env("PROJECT", "runs/improve_train")

    variants = [
        ("yolov8s_dwconv_backbone.yaml", "v1_dwconv_backbone", False),
        ("yolov8s_ca_before_c2f.yaml", "v2_ca_before_c2f", False),
        ("yolov8s_biformer_after_sppf.yaml", "v3_biformer_after_sppf", False),
        ("yolov8s_bifan_neck.yaml", "v4_bifan_neck", False),
        ("yolov8s_siou_loss.yaml", "v5_siou_loss", True),
    ]

    for yaml_name, run_name, use_siou in variants:
        model_yaml = MODEL_DIR / yaml_name
import ultralytics  # noqa: E402


def train_all_variants() -> None:
    data = os.getenv("DATA", "coco8.yaml")
    epochs = int(os.getenv("EPOCHS", "100"))
    imgsz = int(os.getenv("IMGSZ", "640"))
    batch = int(os.getenv("BATCH", "16"))
    device = os.getenv("DEVICE", "0")
    project = os.getenv("PROJECT", "runs/improve_train")

    if "site-packages" in str(Path(ultralytics.__file__).resolve()):
        warnings.warn(
            (
                "Detected site-packages ultralytics import. Continuing anyway; "
                "if custom blocks are missing, run: pip uninstall -y ultralytics && pip install -e ."
            ),
            stacklevel=1,
        )

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
        os.environ["YOLO_USE_SIOU"] = "true" if use_siou else "false"
        model = YOLO(str(model_yaml)).load("yolov8s.pt")
        model.train(
            data=data,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            project=project,
            name=run_name,
        )


# ===================== 主程序 =====================
if __name__ == "__main__":
    train_all_variants()
if __name__ == "__main__":
    train_all_variants()
if "site-packages" in str(Path(ultralytics.__file__).resolve()):
    warnings.warn(
        "Detected site-packages ultralytics import. Attempting to continue, but custom modules (DWConvBlock/CoordAtt/"
        "BiFormerBlock/BiFANFusion) may be missing. If parsing fails, run: pip uninstall -y ultralytics && "
        "pip install -e .",
        stacklevel=1,
    raise RuntimeError(
        "Detected site-packages ultralytics import. Please run from the local repo with editable install: "
        "pip uninstall -y ultralytics && pip install -e ."
    )

VARIANTS = [
    (MODEL_DIR / "yolov8s_dwconv_backbone.yaml", "v1_dwconv_backbone", False),
    (MODEL_DIR / "yolov8s_ca_before_c2f.yaml", "v2_ca_before_c2f", False),
    (MODEL_DIR / "yolov8s_biformer_after_sppf.yaml", "v3_biformer_after_sppf", False),
    (MODEL_DIR / "yolov8s_bifan_neck.yaml", "v4_bifan_neck", False),
    (MODEL_DIR / "yolov8s_siou_loss.yaml", "v5_siou_loss", True),
]

for model_yaml, run_name, use_siou in VARIANTS:
    if not model_yaml.exists():
        raise FileNotFoundError(f"Model YAML not found: {model_yaml}")
    os.environ["YOLO_USE_SIOU"] = "true" if use_siou else "false"
    model = YOLO(str(model_yaml)).load("yolov8s.pt")

VARIANTS = [
    ("experiments/models/yolov8s_dwconv_backbone.yaml", "v1_dwconv_backbone", False),
    ("experiments/models/yolov8s_ca_before_c2f.yaml", "v2_ca_before_c2f", False),
    ("experiments/models/yolov8s_biformer_after_sppf.yaml", "v3_biformer_after_sppf", False),
    ("experiments/models/yolov8s_bifan_neck.yaml", "v4_bifan_neck", False),
    ("experiments/models/yolov8s_siou_loss.yaml", "v5_siou_loss", True),
]

for model_yaml, run_name, use_siou in VARIANTS:
    os.environ["YOLO_USE_SIOU"] = "true" if use_siou else "false"
    model = YOLO(model_yaml).load("yolov8s.pt")
    model.train(
        data=DATA,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        project=PROJECT,
        name=run_name,
    )
