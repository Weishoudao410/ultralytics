# YOLOv8s 五个单改版本（直接源码改造）

运行：

```bash
python experiments/train_scripts/train_variants.py
```

可通过环境变量覆盖：

```bash
DATA=your_dataset.yaml EPOCHS=100 IMGSZ=640 BATCH=16 DEVICE=0 python experiments/train_scripts/train_variants.py
```

输出权重：
- `runs/improve_train/v1_dwconv_backbone/weights/best.pt`
- `runs/improve_train/v2_ca_before_c2f/weights/best.pt`
- `runs/improve_train/v3_biformer_after_sppf/weights/best.pt`
- `runs/improve_train/v4_bifan_neck/weights/best.pt`
- `runs/improve_train/v5_siou_loss/weights/best.pt`
