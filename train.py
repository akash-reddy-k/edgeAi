"""
Fine-tuning — Phase 3 baseline, and Phase 6b architecture-ablation training.

Two modes, selected by --variant:

  (default)         Phase 3: fine-tune pretrained YOLOv8n (COCO weights) on
                     the pseudo-labeled ShanghaiTech dataset. Domain
                     adaptation, not architecture research.

  --variant w125     Phase 6b Stage 2: train one of the two flash-matched
  --variant p34      finalists identified in Phase 6a —
                         w125  configs/yolov8-scaled.yaml scale 'p'  (P3/P4/P5, width 0.125)
                         p34   configs/yolov8-p34.yaml    scale 'n'  (P3/P4,    width 0.25)
                     These land within 5.9% of the same flash budget but
                     spend it differently. See README "The Stage 2 experiment"
                     and ablate.py for why this pair, specifically.

                     Both variants differ from stock YOLOv8n's channel widths
                     and/or head topology. p34 happens to share the stock
                     backbone's shape up through P4 (same width, 0.25), so
                     COCO-pretrained weights would partially transfer for it
                     but not for w125 (uniform width 0.125 matches nothing in
                     stock). Loading pretrained weights for one variant and
                     not the other would hand p34 a head start that has
                     nothing to do with the architecture question Stage 2 is
                     asking. So both train from random initialisation
                     (pretrained=False) to keep the comparison clean.

                     Consequence: absolute accuracy here is NOT comparable to
                     the Phase 3 baseline above, which fine-tunes from COCO.
                     Only the w125-vs-p34 relative comparison is the Stage 2
                     claim — the same pattern as the Phase 3/4 pseudo-label
                     caveat, where the delta is trustworthy and the absolute
                     number is not the point.

Usage:
    python train.py
    python train.py --epochs 30 --batch 32
    python train.py --variant p34  --epochs 150 --patience 30
    python train.py --variant w125 --epochs 150 --patience 30
"""
import argparse
import shutil
from pathlib import Path

import yaml
from ultralytics import YOLO

# The two Phase 6a finalists. Keys/configs/scales/labels match ablate.py's
# VARIANTS exactly, so phase6_ablation.json (cost) and this script's metrics
# (accuracy) describe the same two architectures.
STAGE2_VARIANTS = {
    "w125": ("configs/yolov8-scaled.yaml", "p", "P3/P4/P5  w0.125"),
    "p34":  ("configs/yolov8-p34.yaml",    "n", "P3/P4     w0.25"),
}

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--batch",  type=int, default=16)
parser.add_argument("--imgsz",  type=int, default=640)
parser.add_argument("--dataset", default="data/yolo_dataset/dataset.yaml")
parser.add_argument("--variant", choices=sorted(STAGE2_VARIANTS), default=None,
                     help="Train a Phase 6b architecture finalist from scratch "
                          "instead of fine-tuning the Phase 3 baseline")
parser.add_argument("--patience", type=int, default=30,
                     help="Early-stop if val mAP50-95 doesn't improve for this many "
                          "epochs (--variant runs only; ignored for the default fine-tune)")
parser.add_argument("--cache-dir", default="models/ablation_cache",
                     help="Where to write the scale-resolved temp yaml (--variant runs only)")
parser.add_argument("--cache", default=False, choices=[False, "ram", "disk"],
                     help="Cache decoded images across epochs (default: off, re-decode from "
                          "disk every epoch). CAUTION with 'ram': this dataset decodes to "
                          "~19GB. On unified-memory machines (Apple Silicon) that competes "
                          "directly with the GPU's own memory pool and can exhaust the "
                          "system — confirmed by hanging a 24GB M4 Pro mid-run. Safe on a "
                          "machine with a discrete GPU and enough system RAM to spare.")
parser.add_argument("--device", default="mps",
                     help="mps (Apple Silicon), 0 (first CUDA GPU, e.g. Colab/Kaggle), or cpu")
args = parser.parse_args()

dataset_yaml = Path(args.dataset)
if not dataset_yaml.exists():
    print(f"Dataset not found: {dataset_yaml}")
    print("Run prepare_dataset.py first.")
    raise SystemExit(1)


def resolve_cfg(cfg_path, scale, workdir):
    """Bake the requested scale in as the sole entry of a renamed temp yaml.

    Mirrors ablate.py's resolve_cfg. Ultralytics resolves a model's scale
    from the *filename* via a regex that only recognises [nslmx], so the
    custom narrow rungs ('t', 'p') can't be selected that way. Collapsing
    `scales` to a single 'n' entry and naming the file yolov8n-*.yaml makes
    the selection unambiguous regardless of how the loader guesses.
    """
    base = yaml.safe_load(open(cfg_path))
    d = dict(base)
    d["scales"] = {"n": base["scales"][scale]}
    stem = f"yolov8n-{Path(cfg_path).stem}-{scale}"
    Path(workdir).mkdir(parents=True, exist_ok=True)
    out = Path(workdir) / f"{stem}.yaml"
    yaml.safe_dump(d, open(out, "w"), sort_keys=False)
    return out


if args.variant:
    cfg_path, scale, label = STAGE2_VARIANTS[args.variant]
    resolved = resolve_cfg(cfg_path, scale, args.cache_dir)
    model = YOLO(str(resolved))
    run_name = f"yolov8_ablation_{args.variant}"
    weights_dest = Path(f"models/yolov8_ablation_{args.variant}.pt")
    pretrained = False
    print(f"Phase 6b Stage 2 — training {label}  (key={args.variant})  from random init")
    print(f"  epochs={args.epochs}  patience={args.patience}  batch={args.batch}  "
          f"imgsz={args.imgsz}  device={args.device}\n")
else:
    model = YOLO("models/yolov8n.pt")
    run_name = "yolov8n_shanghaitech"
    weights_dest = Path("models/yolov8n_finetuned.pt")
    pretrained = True
    print(f"Fine-tuning YOLOv8n  |  epochs={args.epochs}  batch={args.batch}  device={args.device}\n")

train_kwargs = dict(
    data=str(dataset_yaml),
    epochs=args.epochs,
    batch=args.batch,
    imgsz=args.imgsz,
    device=args.device,
    project="results/training",
    name=run_name,
    exist_ok=True,
    verbose=True,
    pretrained=pretrained,
)
if args.variant:
    train_kwargs["patience"] = args.patience
if args.cache:
    train_kwargs["cache"] = args.cache

results = model.train(**train_kwargs)

# Copy best weights to models/ for easy access
best_weights = Path(f"runs/detect/results/training/{run_name}/weights/best.pt")
if best_weights.exists():
    shutil.copy(best_weights, weights_dest)
    print(f"\nBest weights saved → {weights_dest}")

# Print the key metrics
metrics = results.results_dict


def fmt(key):
    v = metrics.get(key)
    return f"{v:.4f}" if isinstance(v, (int, float)) else "n/a"


heading = f"Phase 6b Stage 2 — {args.variant}" if args.variant else "Phase 3 Baseline"
init_note = "random init" if args.variant else "fine-tuned, FP32"
print(f"\n--- {heading} metrics ({init_note}) ---")
print(f"  mAP50      : {fmt('metrics/mAP50(B)')}")
print(f"  mAP50-95   : {fmt('metrics/mAP50-95(B)')}")
print(f"  Precision  : {fmt('metrics/precision(B)')}")
print(f"  Recall     : {fmt('metrics/recall(B)')}")

if args.variant:
    print("\nTrain the other Stage 2 variant and compare — see README Phase 6.")
else:
    print("\nRun eval.py after INT8 quantization (Phase 4) and compare these numbers.")
