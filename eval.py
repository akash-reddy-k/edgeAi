"""
Baseline detection evaluation — Phase 2.

Runs YOLOv8n (FP32) on the ShanghaiTech frame directories and computes
per-scene detection statistics split by anomaly vs normal frames.
These numbers are the FP32 baseline — run again after quantization to
measure the accuracy drop (your core Phase 4 research result).

Label format (SHANGHAI_test.txt):
    scene_path  total_frames  has_anomaly  anomaly_start  anomaly_end

Usage:
    python eval.py data/SHANGHAI_Test/
    python eval.py data/SHANGHAI_Test/ --output results/baseline.json --stride 3
"""
import argparse
import json
import os
from pathlib import Path

from ultralytics import YOLO

parser = argparse.ArgumentParser(description="Baseline person-detection evaluation on ShanghaiTech")
parser.add_argument("dataset", help="Path to SHANGHAI_Test directory")
parser.add_argument("--output", default="results/baseline.json",
                    help="Output JSON path (default: results/baseline.json)")
parser.add_argument("--stride", type=int, default=5,
                    help="Sample every Nth frame (default: 5)")
args = parser.parse_args()

dataset_dir = Path(args.dataset)
frames_dir = dataset_dir / "frames"
label_file = dataset_dir / "SHANGHAI_test.txt"

if not label_file.exists():
    print(f"Label file not found: {label_file}")
    raise SystemExit(1)

model = YOLO("models/yolov8n.pt")

# --- Parse label file ---
# Format: scene_path  total_frames  has_anomaly  anomaly_start  anomaly_end
labels = {}
for line in label_file.read_text().splitlines():
    parts = line.strip().split()
    if len(parts) != 5:
        continue
    scene_name = Path(parts[0]).name   # e.g. "01_0015"
    labels[scene_name] = {
        "total_frames": int(parts[1]),
        "has_anomaly": parts[2] == "1",
        "anomaly_start": int(parts[3]),
        "anomaly_end": int(parts[4]),
    }

# Only process scenes that are actually on disk
available_scenes = sorted(p.name for p in frames_dir.iterdir() if p.is_dir())
print(f"Found {len(available_scenes)} scenes on disk, {len(labels)} in label file")
print(f"Model: yolov8n FP32  |  stride: every {args.stride} frames\n")

# --- Evaluate each scene ---
scene_results = []

for scene_name in available_scenes:
    scene_dir = frames_dir / scene_name
    meta = labels.get(scene_name)
    if meta is None:
        print(f"  {scene_name}: no label entry, skipping")
        continue

    # Collect and sort frames by numeric index
    frame_files = sorted(
        scene_dir.glob("*.jpg"),
        key=lambda p: int(p.stem)
    )
    sampled = frame_files[::args.stride]

    anomaly_start = meta["anomaly_start"]
    anomaly_end = meta["anomaly_end"]

    # Counters split by frame type
    counts = {
        "anomaly": {"sampled": 0, "detected": 0, "conf_sum": 0.0},
        "normal":  {"sampled": 0, "detected": 0, "conf_sum": 0.0},
    }
    total_detections = 0

    print(f"  {scene_name} ({'ANOMALY' if meta['has_anomaly'] else 'normal':7s}) "
          f"{len(sampled):4d} sampled frames ...", end=" ", flush=True)

    for frame_path, result in zip(sampled, model(
        [str(f) for f in sampled], stream=True, classes=[0], verbose=False
    )):
        frame_idx = int(frame_path.stem)
        in_anomaly = meta["has_anomaly"] and anomaly_start <= frame_idx <= anomaly_end
        bucket = "anomaly" if in_anomaly else "normal"

        counts[bucket]["sampled"] += 1
        detected = len(result.boxes) > 0
        if detected:
            counts[bucket]["detected"] += 1
            counts[bucket]["conf_sum"] += float(result.boxes.conf.mean())
            total_detections += len(result.boxes)

    def rate(b):
        s = counts[b]["sampled"]
        return round(counts[b]["detected"] / s, 4) if s else None

    def avg_conf(b):
        d = counts[b]["detected"]
        return round(counts[b]["conf_sum"] / d, 4) if d else None

    scene_results.append({
        "scene": scene_name,
        "has_anomaly": meta["has_anomaly"],
        "anomaly_start_frame": anomaly_start if meta["has_anomaly"] else None,
        "anomaly_end_frame": anomaly_end if meta["has_anomaly"] else None,
        "total_frames": meta["total_frames"],
        "sampled_frames": len(sampled),
        "stride": args.stride,
        "anomaly_frames_sampled": counts["anomaly"]["sampled"],
        "normal_frames_sampled": counts["normal"]["sampled"],
        "detection_rate_anomaly": rate("anomaly"),
        "detection_rate_normal": rate("normal"),
        "avg_confidence_anomaly": avg_conf("anomaly"),
        "avg_confidence_normal": avg_conf("normal"),
        "total_person_detections": total_detections,
    })

    dr_a = rate("anomaly")
    dr_n = rate("normal")
    print(f"anomaly det={dr_a if dr_a is not None else 'n/a':>6}  normal det={dr_n if dr_n is not None else 'n/a':>6}")

# --- Aggregate summary ---
anomaly_scenes = [r for r in scene_results if r["has_anomaly"]]
normal_scenes  = [r for r in scene_results if not r["has_anomaly"]]

def mean(vals):
    vals = [v for v in vals if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None

summary = {
    "model": "yolov8n-FP32",
    "stride": args.stride,
    "total_scenes_evaluated": len(scene_results),
    "anomaly_scenes": len(anomaly_scenes),
    "normal_scenes": len(normal_scenes),
    "avg_detection_rate_in_anomaly_frames": mean(r["detection_rate_anomaly"] for r in anomaly_scenes),
    "avg_detection_rate_in_normal_frames":  mean(
        r["detection_rate_normal"] for r in scene_results
    ),
    "avg_confidence_anomaly_frames": mean(r["avg_confidence_anomaly"] for r in anomaly_scenes),
    "avg_confidence_normal_frames":  mean(
        r["avg_confidence_normal"] for r in scene_results if r["avg_confidence_normal"]
    ),
}

output = {"summary": summary, "scenes": scene_results}

os.makedirs(Path(args.output).parent, exist_ok=True)
with open(args.output, "w") as f:
    json.dump(output, f, indent=2)

print(f"\n--- Summary ({summary['model']}) ---")
print(f"  Scenes evaluated : {summary['total_scenes_evaluated']}  "
      f"({summary['anomaly_scenes']} anomaly, {summary['normal_scenes']} normal)")
print(f"  Detection rate — anomaly frames : {summary['avg_detection_rate_in_anomaly_frames']}")
print(f"  Detection rate — normal frames  : {summary['avg_detection_rate_in_normal_frames']}")
print(f"  Avg confidence  — anomaly frames: {summary['avg_confidence_anomaly_frames']}")
print(f"  Avg confidence  — normal frames : {summary['avg_confidence_normal_frames']}")
print(f"\nResults saved → {args.output}")
print("Run again after INT8 quantization (Phase 4) and diff the numbers.")
