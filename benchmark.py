"""
Phase 4 — Benchmark.

Measures the accuracy / size / latency tradeoff across three variants of the
Phase 3 fine-tuned model:

    PyTorch FP32   models/yolov8n_finetuned.pt
    ONNX FP32      models/yolov8n_finetuned_fp32.onnx
    ONNX INT8      models/yolov8n_finetuned_int8.onnx

This table is the core research output of the project — the cost of compression,
measured rather than assumed.

Fairness controls:
    - Same val split, same labels, for all three arms.
    - batch=1 everywhere (the ONNX graphs are exported with a static batch of 1,
      so this is forced for ONNX and matched on PyTorch for comparability).
    - rect=False so every image is letterboxed to a square 640x640. Ultralytics'
      default rectangular batching would feed the PyTorch arm different input
      shapes than the static-shape ONNX arms and quietly skew the mAP delta.
    - Latency measured on CPU deliberately. The Phase 5 target is a plain MCU
      with no AI accelerator, so MPS/CUDA numbers would be misleading.

Absolute mAP is self-referential (val labels are pseudo-labels from the base
model — see README "Methodology & Limitations"). The DELTA between arms is the
valid measurement, because the label set is held constant across all three.

Usage:
    python benchmark.py
    python benchmark.py --latency-images 200
    python benchmark.py --skip-accuracy        # latency + size only, fast
"""
import argparse
import json
import statistics
import time
from pathlib import Path

from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="data/yolo_dataset/dataset.yaml")
parser.add_argument("--models-dir", default="models")
parser.add_argument("--imgsz", type=int, default=640)
parser.add_argument("--device", default="cpu",
                    help="Device for both val and latency (default: cpu — the honest edge proxy)")
parser.add_argument("--latency-images", type=int, default=100,
                    help="Number of val images to time (default: 100)")
parser.add_argument("--warmup", type=int, default=10,
                    help="Untimed warmup inferences before measuring (default: 10)")
parser.add_argument("--skip-accuracy", action="store_true",
                    help="Skip the mAP pass (slow on CPU); report size + latency only")
parser.add_argument("--output", default="results/phase4_benchmark.json")
args = parser.parse_args()

MODELS = Path(args.models_dir)
DATASET = Path(args.dataset)

VARIANTS = [
    ("PyTorch FP32", MODELS / "yolov8n_finetuned.pt"),
    ("ONNX FP32",    MODELS / "yolov8n_finetuned_fp32.onnx"),
    ("ONNX INT8",    MODELS / "yolov8n_finetuned_int8.onnx"),
]

missing = [str(p) for _, p in VARIANTS if not p.exists()]
if missing:
    print("Missing model artifacts:\n  " + "\n  ".join(missing))
    print("\nRun quantize.py first.")
    raise SystemExit(1)

if not DATASET.exists():
    print(f"Dataset not found: {DATASET}\nRun prepare_dataset.py first.")
    raise SystemExit(1)

# ---------------------------------------------------------------- latency images
# Evenly spaced across the val split so the timing sample spans every scene
# rather than sitting inside one clip.
val_images = sorted((DATASET.parent / "images" / "val").rglob("*.jpg"))
if not val_images:
    print(f"No val images found under {DATASET.parent / 'images' / 'val'}")
    raise SystemExit(1)

step = max(1, len(val_images) // args.latency_images)
timing_images = [str(p) for p in val_images[::step][:args.latency_images]]


def measure_latency(model_path):
    """Per-image inference latency, warmed up, on single images (batch=1).

    Reports ultralytics' own speed breakdown so pre/post-processing is visible
    separately — on an MCU those stages are not free either.
    """
    model = YOLO(str(model_path), task="detect")

    for img in timing_images[:args.warmup]:
        model.predict(img, imgsz=args.imgsz, device=args.device, verbose=False)

    pre, inf, post = [], [], []
    wall_start = time.perf_counter()
    for img in timing_images:
        result = model.predict(img, imgsz=args.imgsz, device=args.device, verbose=False)[0]
        pre.append(result.speed["preprocess"])
        inf.append(result.speed["inference"])
        post.append(result.speed["postprocess"])
    wall = (time.perf_counter() - wall_start) * 1000 / len(timing_images)

    return {
        "n_images": len(timing_images),
        "preprocess_ms": round(statistics.mean(pre), 2),
        # Median over mean: CPU scheduling spikes skew the mean, and the
        # question "how long does one frame take" is a typical-case question.
        "inference_ms_median": round(statistics.median(inf), 2),
        "inference_ms_mean": round(statistics.mean(inf), 2),
        "inference_ms_stdev": round(statistics.stdev(inf), 2) if len(inf) > 1 else 0.0,
        "postprocess_ms": round(statistics.mean(post), 2),
        "end_to_end_ms": round(wall, 2),
        "fps_end_to_end": round(1000 / wall, 2),
    }


def measure_accuracy(model_path):
    model = YOLO(str(model_path), task="detect")
    metrics = model.val(
        data=str(DATASET),
        imgsz=args.imgsz,
        batch=1,      # ONNX graphs are static batch-1; matched on PyTorch for fairness
        rect=False,   # square letterbox everywhere — see module docstring
        device=args.device,
        verbose=False,
        plots=False,
    )
    return {
        "mAP50":    round(float(metrics.box.map50), 4),
        "mAP50-95": round(float(metrics.box.map),   4),
        "precision": round(float(metrics.box.mp),   4),
        "recall":    round(float(metrics.box.mr),   4),
    }


# ---------------------------------------------------------------- run
print(f"Benchmarking 3 variants  |  device={args.device}  imgsz={args.imgsz}")
print(f"  accuracy : {'SKIPPED' if args.skip_accuracy else f'{DATASET} ({len(val_images)} val images)'}")
print(f"  latency  : {len(timing_images)} images, {args.warmup} warmup\n")

results = {}
for label, path in VARIANTS:
    print(f"--- {label} ---")
    entry = {"path": str(path), "size_mb": round(path.stat().st_size / (1024 * 1024), 2)}

    if not args.skip_accuracy:
        print("  measuring accuracy ...")
        entry["accuracy"] = measure_accuracy(path)

    print("  measuring latency ...")
    entry["latency"] = measure_latency(path)
    results[label] = entry
    print()

# ---------------------------------------------------------------- table
BASELINE = "PyTorch FP32"
base = results[BASELINE]

print("\n=== Phase 4 — Compression Tradeoff ===\n")

header = f"{'Variant':<14} {'Size MB':>8} {'vs base':>8}"
if not args.skip_accuracy:
    header += f" {'mAP50':>8} {'mAP50-95':>9} {'Δ mAP50':>9}"
header += f" {'Infer ms':>9} {'Speedup':>8}"
print(header)
print("-" * len(header))

for label, entry in results.items():
    size = entry["size_mb"]
    inf = entry["latency"]["inference_ms_median"]
    row = f"{label:<14} {size:>8.2f} {size / base['size_mb']:>7.2f}x"
    if not args.skip_accuracy:
        acc = entry["accuracy"]
        d_map = acc["mAP50"] - base["accuracy"]["mAP50"]
        row += f" {acc['mAP50']:>8.4f} {acc['mAP50-95']:>9.4f} {d_map:>+9.4f}"
    row += f" {inf:>9.2f} {base['latency']['inference_ms_median'] / inf:>7.2f}x"
    print(row)

print(f"\n(baseline = {BASELINE}; speedup >1 is faster, size <1x is smaller)")

# The ONNX-to-ONNX comparison is the apples-to-apples compression number.
# PyTorch .pt vs ONNX size is not a like-for-like measure: the .pt is a
# pickled state_dict, the ONNX carries a full serialised graph.
fp32_onnx, int8_onnx = results["ONNX FP32"], results["ONNX INT8"]
print("\n--- Quantization effect (ONNX FP32 → ONNX INT8, like-for-like) ---")
print(f"  Size      : {fp32_onnx['size_mb']:.2f} MB → {int8_onnx['size_mb']:.2f} MB  "
      f"({(1 - int8_onnx['size_mb'] / fp32_onnx['size_mb']) * 100:.1f}% smaller)")
print(f"  Latency   : {fp32_onnx['latency']['inference_ms_median']:.2f} ms → "
      f"{int8_onnx['latency']['inference_ms_median']:.2f} ms")
if not args.skip_accuracy:
    d = int8_onnx["accuracy"]["mAP50"] - fp32_onnx["accuracy"]["mAP50"]
    d95 = int8_onnx["accuracy"]["mAP50-95"] - fp32_onnx["accuracy"]["mAP50-95"]
    print(f"  mAP50     : {fp32_onnx['accuracy']['mAP50']:.4f} → "
          f"{int8_onnx['accuracy']['mAP50']:.4f}  ({d:+.4f})")
    print(f"  mAP50-95  : {fp32_onnx['accuracy']['mAP50-95']:.4f} → "
          f"{int8_onnx['accuracy']['mAP50-95']:.4f}  ({d95:+.4f})")

out_path = Path(args.output)
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps({
    "config": {
        "device": args.device,
        "imgsz": args.imgsz,
        "batch": 1,
        "rect": False,
        "dataset": str(DATASET),
        "val_images": len(val_images),
        "latency_images": len(timing_images),
        "warmup": args.warmup,
        "accuracy_measured": not args.skip_accuracy,
    },
    "note": ("Absolute mAP is self-referential — val labels are pseudo-labels "
             "from the base YOLOv8n. Only the deltas between variants are valid. "
             "See README 'Methodology & Limitations'."),
    "variants": results,
}, indent=2))
print(f"\nSaved → {out_path}")
