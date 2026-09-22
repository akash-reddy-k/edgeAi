"""
Phase 3 — Dataset preparation.

Generates YOLO-format bounding box labels for ShanghaiTech frames using
the pretrained YOLOv8n as a pseudo-labeler (domain adaptation: COCO → campus).
Creates a train/val scene split and symlinks images to avoid duplicating 874 MB.

Output structure:
    data/yolo_dataset/
    ├── images/train/<scene>/  → symlinks to original frames
    ├── images/val/<scene>/    → symlinks to original frames
    ├── labels/train/<scene>/  → generated .txt label files
    ├── labels/val/<scene>/    → generated .txt label files
    └── dataset.yaml

Usage:
    python prepare_dataset.py
    python prepare_dataset.py --conf 0.3  # stricter pseudo-labels
"""
import argparse
import os
from pathlib import Path

from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument("--conf", type=float, default=0.25,
                    help="Confidence threshold for pseudo-label generation (default: 0.25)")
parser.add_argument("--dataset-dir", default="data/SHANGHAI_Test")
parser.add_argument("--output-dir", default="data/yolo_dataset")
args = parser.parse_args()

DATASET_DIR = Path(args.dataset_dir)
OUTPUT_DIR  = Path(args.output_dir)
FRAMES_DIR  = DATASET_DIR / "frames"

# Val scenes chosen to balance anomaly (A) and normal (N) across both splits
VAL_SCENES = {
    "01_0028",   # A
    "01_0055",   # A
    "01_0139",   # A
    "01_0177",   # A
    "01_006",    # N
    "01_021",    # N
    "01_028",    # N
    "01_030",    # N
}

model = YOLO("models/yolov8n.pt")

scene_dirs = sorted(p for p in FRAMES_DIR.iterdir() if p.is_dir())
print(f"Scenes found: {len(scene_dirs)}  |  Val scenes: {len(VAL_SCENES)}\n")

total_frames = 0
total_detections = 0

for scene_dir in scene_dirs:
    scene = scene_dir.name
    split = "val" if scene in VAL_SCENES else "train"

    img_out_dir   = OUTPUT_DIR / "images" / split / scene
    label_out_dir = OUTPUT_DIR / "labels" / split / scene
    img_out_dir.mkdir(parents=True, exist_ok=True)
    label_out_dir.mkdir(parents=True, exist_ok=True)

    frame_files = sorted(scene_dir.glob("*.jpg"), key=lambda p: int(p.stem))
    print(f"  [{split:5s}] {scene:12s}  {len(frame_files):4d} frames", end="  ", flush=True)

    scene_detections = 0

    for frame_path in frame_files:
        # Symlink image (avoids copying 874 MB)
        link = img_out_dir / frame_path.name
        if not link.exists():
            link.symlink_to(frame_path.resolve())

        # Generate pseudo-label
        label_path = label_out_dir / (frame_path.stem + ".txt")
        if label_path.exists():
            continue

        result = model(str(frame_path), classes=[0], conf=args.conf, verbose=False)[0]

        lines = []
        for box in result.boxes:
            xc, yc, w, h = box.xywhn[0].tolist()  # normalised xywh
            lines.append(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

        label_path.write_text("\n".join(lines))
        scene_detections += len(result.boxes)

    total_frames += len(frame_files)
    total_detections += scene_detections
    print(f"detections: {scene_detections}")

# Write dataset.yaml
yaml_path = OUTPUT_DIR / "dataset.yaml"
yaml_path.write_text(f"""\
path: {OUTPUT_DIR.resolve()}
train: images/train
val: images/val
nc: 1
names: ['person']
""")

print(f"\nDataset ready → {OUTPUT_DIR}/")
print(f"  Total frames  : {total_frames:,}")
print(f"  Avg detections: {total_detections / total_frames:.2f} per frame")
print(f"  dataset.yaml  : {yaml_path}")
print(f"\nNext: python train.py")
