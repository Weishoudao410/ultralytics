from __future__ import annotations

import os
import sys
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


def _register_custom_modules_for_runtime() -> None:
    """Register local experimental modules into whichever ultralytics package is imported at runtime."""
    from ultralytics.nn import tasks as nn_tasks
    from ultralytics.nn.modules import BiFANFusion, BiFormerBlock, CoordAtt, DWConvBlock

    # Make parse_model globals()[m] resolvable even when ultralytics comes from site-packages.
    nn_tasks.DWConvBlock = DWConvBlock
    nn_tasks.CoordAtt = CoordAtt
    nn_tasks.BiFormerBlock = BiFormerBlock
    nn_tasks.BiFANFusion = BiFANFusion


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return default if value is None else value


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
        if not model_yaml.exists():
            raise FileNotFoundError(f"Model YAML not found: {model_yaml}")

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


if __name__ == "__main__":
    train_all_variants()
