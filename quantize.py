"""
Phase 4 — Quantization.

Exports the Phase 3 fine-tuned model to ONNX FP32 and ONNX INT8.

Calibration methodology:
    INT8 quantization needs representative images to calibrate activation
    ranges. Ultralytics draws these from the `val:` split of whatever dataset
    yaml it is given — so passing the real dataset.yaml would calibrate on the
    same images used for evaluation (calibration leakage).

    Instead this script builds a dedicated calibration set by subsampling the
    TRAIN split, and writes a yaml whose `val:` points at it. Calibration and
    evaluation therefore use disjoint data.

Usage:
    python quantize.py
    python quantize.py --calib-size 2000
"""
import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="models/yolov8n_finetuned.pt")
parser.add_argument("--calib-size", type=int, default=1000,
                    help="Number of calibration images to sample from train split (default: 1000)")
parser.add_argument("--dataset-dir", default="data/yolo_dataset")
args = parser.parse_args()

DATASET = Path(args.dataset_dir)
MODEL   = Path(args.model)

if not MODEL.exists():
    print(f"Model not found: {MODEL}\nRun train.py first.")
    raise SystemExit(1)

# ---------------------------------------------------------------- calibration set
# Subsample the train split. Ultralytics' calibration dataloader scans labels
# alongside images, so both are symlinked.
calib_img_dir   = DATASET / "images" / "calib"
calib_label_dir = DATASET / "labels" / "calib"

if calib_img_dir.exists():
    shutil.rmtree(calib_img_dir)
if calib_label_dir.exists():
    shutil.rmtree(calib_label_dir)
calib_img_dir.mkdir(parents=True)
calib_label_dir.mkdir(parents=True)

train_images = sorted((DATASET / "images" / "train").rglob("*.jpg"))
if not train_images:
    print(f"No train images found under {DATASET / 'images' / 'train'}")
    raise SystemExit(1)

# Evenly spaced sample across all scenes rather than the first N (which would
# only cover the alphabetically-first scenes).
step = max(1, len(train_images) // args.calib_size)
sampled = train_images[::step][:args.calib_size]

for img in sampled:
    scene = img.parent.name
    label_src = DATASET / "labels" / "train" / scene / (img.stem + ".txt")

    # Flatten into one dir, prefixing scene to keep names unique
    flat_name = f"{scene}_{img.stem}"
    (calib_img_dir / f"{flat_name}.jpg").symlink_to(img.resolve())
    if label_src.exists():
        (calib_label_dir / f"{flat_name}.txt").symlink_to(label_src.resolve())

calib_yaml = DATASET / "calibration.yaml"
calib_yaml.write_text(f"""\
# Calibration-only dataset. `val:` deliberately points at a subsample of the
# TRAIN split so INT8 calibration never sees the evaluation images.
path: {DATASET.resolve()}
train: images/calib
val: images/calib
nc: 1
names: ['person']
""")

print(f"Calibration set: {len(sampled)} images sampled from {len(train_images)} train images")
print(f"  → {calib_img_dir}\n")

# ---------------------------------------------------------------- exports
def size_mb(p):
    return Path(p).stat().st_size / (1024 * 1024)

out = {}

print("Exporting ONNX FP32 ...")
fp32_src = Path(YOLO(str(MODEL)).export(format="onnx"))
# Rename immediately: the INT8 export below writes its own intermediate to this
# same path and deletes it on cleanup, which would take the FP32 file with it.
out["fp32"] = MODEL.parent / f"{MODEL.stem}_fp32.onnx"
shutil.move(str(fp32_src), out["fp32"])

print("\nExporting ONNX INT8 ...")
int8_src = Path(YOLO(str(MODEL)).export(format="onnx", quantize="int8", data=str(calib_yaml)))
out["int8"] = MODEL.parent / f"{MODEL.stem}_int8.onnx"
if int8_src.resolve() != out["int8"].resolve():
    shutil.move(str(int8_src), out["int8"])

print("\n--- Export summary ---")
print(f"  PyTorch FP32 : {size_mb(MODEL):6.2f} MB   {MODEL}")
print(f"  ONNX FP32    : {size_mb(out['fp32']):6.2f} MB   {out['fp32']}")
print(f"  ONNX INT8    : {size_mb(out['int8']):6.2f} MB   {out['int8']}")
print(f"\n  Size reduction (PyTorch → INT8): "
      f"{(1 - size_mb(out['int8']) / size_mb(MODEL)) * 100:.1f}%")
print("\nNext: python benchmark.py")
