from __future__ import annotations

import os
from ultralytics import YOLO

DATA = os.getenv("DATA", "coco8.yaml")
EPOCHS = int(os.getenv("EPOCHS", "100"))
IMGSZ = int(os.getenv("IMGSZ", "640"))
BATCH = int(os.getenv("BATCH", "16"))
DEVICE = os.getenv("DEVICE", "0")
PROJECT = os.getenv("PROJECT", "runs/improve_train")

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
