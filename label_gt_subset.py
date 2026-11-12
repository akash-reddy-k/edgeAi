"""
Ground-truth subset preparation — the accuracy anchor.

Every mAP figure elsewhere in this project is self-referential: validation
labels are pseudo-labels from the same base model being fine-tuned (see
README "Methodology & Limitations"). That section already names the cheap
fix: hand-annotate ~200 validation frames to anchor the relative numbers to
one honest absolute measurement. This script does the mechanical half of
that — it does NOT label anything for you.

What it does:
    1. Samples ~200 frames evenly across the 8 val scenes (so the subset
       keeps the same 4-anomaly / 4-normal balance as the full val split).
    2. Copies each sampled frame's existing pseudo-label as a STARTING POINT
       — correcting an approximately-right box is faster than drawing one
       from scratch, and the pseudo-labeler is usually not far off for
       in-frame, unoccluded people. Frames with no pseudo-label get an empty
       .txt (correct from blank).
    3. Writes a dataset.yaml pointing at the subset, so eval_gt_subset.py
       can score it the moment the boxes are corrected.

What you do in between running this and running eval_gt_subset.py:
    Open data/gt_subset/images/val/<scene>/ in a YOLO-format box editor
    (labelImg, CVAT, Label Studio — any of them read/write this layout) and
    correct data/gt_subset/labels/val/<scene>/*.txt in place. Class 0 = person.

Usage:
    python label_gt_subset.py
    python label_gt_subset.py --n-per-scene 30   # ~240 frames instead of ~200
"""
import argparse
import json
import shutil
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--dataset-dir", default="data/yolo_dataset",
                    help="Existing prepare_dataset.py output (pseudo-labels + val split)")
parser.add_argument("--output-dir", default="data/gt_subset")
parser.add_argument("--n-per-scene", type=int, default=25,
                    help="Frames sampled per val scene, evenly spaced (default: 25 -> ~200 over 8 scenes)")
args = parser.parse_args()

DATASET_DIR = Path(args.dataset_dir)
OUTPUT_DIR = Path(args.output_dir)
VAL_IMAGES_DIR = DATASET_DIR / "images" / "val"
VAL_LABELS_DIR = DATASET_DIR / "labels" / "val"

# Same split prepare_dataset.py uses — kept here so the subset's anomaly/normal
# balance is reportable without re-deriving it from the frame label file.
ANOMALY_VAL_SCENES = {"01_0028", "01_0055", "01_0139", "01_0177"}

if not VAL_IMAGES_DIR.exists():
    print(f"{VAL_IMAGES_DIR} not found. Run prepare_dataset.py first.")
    raise SystemExit(1)

scene_dirs = sorted(p for p in VAL_IMAGES_DIR.iterdir() if p.is_dir())
if not scene_dirs:
    print(f"No scenes under {VAL_IMAGES_DIR}.")
    raise SystemExit(1)

manifest = {"n_per_scene": args.n_per_scene, "scenes": {}}
total_sampled = 0
total_with_starting_box = 0

print(f"Sampling ~{args.n_per_scene} frames/scene from {len(scene_dirs)} val scenes ...\n")

for scene_dir in scene_dirs:
    scene = scene_dir.name
    img_out_dir = OUTPUT_DIR / "images" / "val" / scene
    label_out_dir = OUTPUT_DIR / "labels" / "val" / scene
    img_out_dir.mkdir(parents=True, exist_ok=True)
    label_out_dir.mkdir(parents=True, exist_ok=True)

    frame_files = sorted(scene_dir.glob("*.jpg"), key=lambda p: int(p.stem))
    step = max(1, len(frame_files) // args.n_per_scene)
    sampled = frame_files[::step][:args.n_per_scene]

    with_box = 0
    for frame_path in sampled:
        # frame_path is itself a symlink (prepare_dataset.py's trick to avoid
        # copying 874 MB) — resolve so the subset survives data/yolo_dataset
        # being rebuilt or removed later.
        dst_img = img_out_dir / frame_path.name
        if not dst_img.exists():
            dst_img.symlink_to(frame_path.resolve())

        dst_label = label_out_dir / (frame_path.stem + ".txt")
        if not dst_label.exists():
            src_label = VAL_LABELS_DIR / scene / (frame_path.stem + ".txt")
            if src_label.exists() and src_label.read_text().strip():
                shutil.copy(src_label, dst_label)
                with_box += 1
            else:
                dst_label.touch()

    manifest["scenes"][scene] = {
        "has_anomaly": scene in ANOMALY_VAL_SCENES,
        "sampled": len(sampled),
        "with_starting_box": with_box,
    }
    total_sampled += len(sampled)
    total_with_starting_box += with_box
    print(f"  {scene:12s} {'A' if scene in ANOMALY_VAL_SCENES else 'N'}  "
          f"{len(sampled):3d} frames  ({with_box} with a starting box)")

yaml_path = OUTPUT_DIR / "dataset.yaml"
yaml_path.write_text(f"""\
path: {OUTPUT_DIR.resolve()}
train: images/val
val: images/val
nc: 1
names: ['person']
""")

manifest["total_sampled"] = total_sampled
manifest["total_with_starting_box"] = total_with_starting_box
manifest_path = OUTPUT_DIR / "manifest.json"
manifest_path.write_text(json.dumps(manifest, indent=2))

print(f"\n{total_sampled} frames sampled, {total_with_starting_box} pre-filled with a pseudo-label "
      f"starting box ({total_sampled - total_with_starting_box} blank — no detection to start from).")
print(f"Manifest  -> {manifest_path}")
print(f"Images    -> {OUTPUT_DIR / 'images' / 'val'}")
print(f"Labels    -> {OUTPUT_DIR / 'labels' / 'val'}  (correct these — YOLO format, class 0 = person)")
print(f"\nNext: hand-correct the boxes in a YOLO-format editor (labelImg / CVAT / Label Studio),")
print(f"then run: python eval_gt_subset.py")
