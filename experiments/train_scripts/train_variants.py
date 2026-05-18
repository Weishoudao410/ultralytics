from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

# Resolve repository root from this script path.
ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "experiments" / "models"

# Ensure local repository is preferred in import resolution.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO  # noqa: E402
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
