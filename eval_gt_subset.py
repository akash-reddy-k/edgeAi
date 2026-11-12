"""
Evaluate the hand-corrected ground-truth subset — the one absolute accuracy anchor.

Run this AFTER manually correcting the boxes written by label_gt_subset.py
in data/gt_subset/labels/val/. Every other mAP figure in this project is
self-referential (val labels are pseudo-labels from the base model being
fine-tuned — see README "Methodology & Limitations"); this is the one number
measured against boxes a human actually verified.

This does not replace the FP32-vs-INT8 or architecture-ablation comparisons
elsewhere in the project (results/phase4_benchmark.json,
results/phase6_ablation.json) — those are relative comparisons on held-constant
labels and stay valid regardless of this number. This script exists to anchor
the absolute level those relative deltas sit on top of.

Usage:
    python eval_gt_subset.py
    python eval_gt_subset.py --model models/yolov8_ablation_p34.pt
"""
import argparse
import json
from pathlib import Path

from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="data/gt_subset/dataset.yaml")
parser.add_argument("--model", default="models/yolov8n_finetuned.pt")
parser.add_argument("--imgsz", type=int, default=640)
parser.add_argument("--output", default="results/gt_subset_eval.json")
args = parser.parse_args()

dataset_path = Path(args.dataset)
if not dataset_path.exists():
    print(f"{dataset_path} not found.\n"
          f"Run label_gt_subset.py first, then hand-correct the boxes it writes "
          f"before running this.")
    raise SystemExit(1)

# Ultralytics validates its label cache by total file SIZE plus paths, not
# content (ultralytics.data.utils.get_hash). YOLO label lines are fixed-width,
# so nudging a box in an editor leaves every file the same size — a stale cache
# would then silently score the original pseudo-labels as the corrected ones.
for cache in (dataset_path.parent / "labels").rglob("*.cache"):
    cache.unlink()

model = YOLO(args.model, task="detect")
metrics = model.val(
    data=str(dataset_path),
    imgsz=args.imgsz,
    batch=1,     # matched to the benchmark.py convention — see its docstring
    rect=False,
    verbose=False,
    plots=False,
)

result = {
    "model": args.model,
    "dataset": str(dataset_path),
    "note": ("Ground truth here is hand-corrected (see label_gt_subset.py), not "
             "pseudo-labelled. This is the project's one absolute-accuracy anchor; "
             "every other mAP figure is relative-only — see README "
             "'Methodology & Limitations'."),
    "mAP50": round(float(metrics.box.map50), 4),
    "mAP50-95": round(float(metrics.box.map), 4),
    "precision": round(float(metrics.box.mp), 4),
    "recall": round(float(metrics.box.mr), 4),
}

out_path = Path(args.output)
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(result, indent=2))

print(json.dumps(result, indent=2))
print(f"\nSaved -> {out_path}")
