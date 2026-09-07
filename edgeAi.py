import argparse
import cv2
from collections import defaultdict
from ultralytics import YOLO

parser = argparse.ArgumentParser(description="Loitering detection — Edge AI Phase 2")
parser.add_argument("video", help="Path to video file (e.g. files/TwoKids.mp4)")
parser.add_argument("--loiter-seconds", type=float, default=5.0,
                    help="Seconds a person must remain in frame to trigger an alert (default: 5)")
args = parser.parse_args()

model = YOLO("models/yolov8n.pt")

cap = cv2.VideoCapture(args.video)
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
cap.release()

frame_threshold = args.loiter_seconds * fps

frames_seen = defaultdict(int)  # person_id -> consecutive frames in scene
alerted_ids = set()

def send_alert(person_id, seconds):
    # Placeholder — swap for a real Telegram/webhook call
    print(f"ALERT: Person #{person_id} loitering — present for {seconds:.1f}s")

print(f"Running loitering detection on: {args.video}  (threshold: {args.loiter_seconds}s @ {fps:.1f} fps)")

# persist=True keeps tracker state across frames so IDs stay consistent
for result in model.track(args.video, stream=True, classes=[0], verbose=False, persist=True):
    if result.boxes.id is None:
        continue

    current_ids = set(result.boxes.id.int().tolist())

    for pid in current_ids:
        frames_seen[pid] += 1
        if frames_seen[pid] >= frame_threshold and pid not in alerted_ids:
            send_alert(pid, frames_seen[pid] / fps)
            alerted_ids.add(pid)

    # Reset counter for IDs that left the frame
    for pid in list(frames_seen):
        if pid not in current_ids:
            frames_seen[pid] = 0

if not alerted_ids:
    print("No loitering detected.")
