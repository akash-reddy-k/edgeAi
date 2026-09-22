"""
Phase 3 — Fine-tuning.

Fine-tunes YOLOv8n on the pseudo-labeled ShanghaiTech dataset (campus domain
adaptation). Uses Apple MPS acceleration. Best weights are saved to models/.

Results (mAP50, mAP50-95, precision, recall) are the Phase 3 baseline —
run eval.py after this and again after INT8 quantization (Phase 4) to
measure the accuracy drop.

Usage:
    python train.py
    python train.py --epochs 30 --batch 32
"""
import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--batch",  type=int, default=16)
parser.add_argument("--imgsz",  type=int, default=640)
parser.add_argument("--dataset", default="data/yolo_dataset/dataset.yaml")
args = parser.parse_args()

dataset_yaml = Path(args.dataset)
if not dataset_yaml.exists():
    print(f"Dataset not found: {dataset_yaml}")
    print("Run prepare_dataset.py first.")
    raise SystemExit(1)

model = YOLO("models/yolov8n.pt")

print(f"Fine-tuning YOLOv8n  |  epochs={args.epochs}  batch={args.batch}  device=mps\n")

results = model.train(
    data=str(dataset_yaml),
    epochs=args.epochs,
    batch=args.batch,
    imgsz=args.imgsz,
    device="mps",
    project="results/training",
    name="yolov8n_shanghaitech",
    exist_ok=True,
    verbose=True,
)

# Copy best weights to models/ for easy access
best_weights = Path("runs/detect/results/training/yolov8n_shanghaitech/weights/best.pt")
if best_weights.exists():
    dest = Path("models/yolov8n_finetuned.pt")
    shutil.copy(best_weights, dest)
    print(f"\nBest weights saved → {dest}")

# Print the key metrics
metrics = results.results_dict
print("\n--- Phase 3 Baseline Metrics (fine-tuned, FP32) ---")
print(f"  mAP50      : {metrics.get('metrics/mAP50(B)',    'n/a'):.4f}")
print(f"  mAP50-95   : {metrics.get('metrics/mAP50-95(B)', 'n/a'):.4f}")
print(f"  Precision  : {metrics.get('metrics/precision(B)', 'n/a'):.4f}")
print(f"  Recall     : {metrics.get('metrics/recall(B)',    'n/a'):.4f}")
print("\nRun eval.py after INT8 quantization (Phase 4) and compare these numbers.")
