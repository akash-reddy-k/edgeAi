import argparse
import cv2
from collections import defaultdict
from ultralytics import YOLO

parser = argparse.ArgumentParser(description="Loitering detection — Edge AI Phase 2")
parser.add_argument("video", help="Path to video file (e.g. files/TwoKids.mp4)")
parser.add_argument("--loiter-seconds", type=float, default=5.0,
                    help="Seconds a person must remain in frame to trigger a loitering alert (default: 5)")
parser.add_argument("--owners-away", action="store_true", default=True,
                    help="Enable alert mode — house owners are away (default: True)")
args = parser.parse_args()

model = YOLO("models/yolov8n.pt")

cap = cv2.VideoCapture(args.video)
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
cap.release()

# frames a person must be tracked continuously before alerting
frame_threshold = args.loiter_seconds * fps

frames_seen = defaultdict(int)   # person_id -> consecutive frames in scene
alerted_ids = set()

def send_alert(person_id, seconds):
    # Placeholder — swap for a real Telegram/webhook call
    print(f"ALERT: Person #{person_id} loitering — present for {seconds:.1f}s")

print(f"Running loitering detection on: {args.video}  (threshold: {args.loiter_seconds}s @ {fps:.1f} fps)")

# persist=True keeps tracker state across frames so IDs stay consistent
results = model.track(args.video, stream=True, classes=[0], verbose=False, persist=True)

for result in results:
    if not args.owners_away or result.boxes.id is None:
        continue

    current_ids = set(result.boxes.id.int().tolist())

    for pid in current_ids:
        frames_seen[pid] += 1
        if frames_seen[pid] >= frame_threshold and pid not in alerted_ids:
            send_alert(pid, frames_seen[pid] / fps)
            alerted_ids.add(pid)

    # Reset counter for any ID that left the frame this tick
    for pid in list(frames_seen):
        if pid not in current_ids:
            frames_seen[pid] = 0

if not alerted_ids:
    print("No loitering detected.")
