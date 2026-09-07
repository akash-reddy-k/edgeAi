import argparse
from ultralytics import YOLO

parser = argparse.ArgumentParser(description="Person detection alert system")
parser.add_argument("video", help="Path to the video file (e.g. files/TwoKids.mp4)")
parser.add_argument("--owners-away", action="store_true", default=True,
                    help="Simulate house owners not home (default: True)")
args = parser.parse_args()

model = YOLO("models/yolov8n.pt")

CONSECUTIVE_FRAMES_THRESHOLD = 5
alert_counter = 0
alert_sent = False

def send_alert(person_count):
    # Placeholder — swap this for a real Telegram/webhook call
    print(f"ALERT: {person_count} person(s) detected while house is away!")

videoframes = model(args.video, stream=True, classes=[0], vid_stride=5, verbose=False)
for frame_result in videoframes:
    person_count = len(frame_result.boxes)

    if args.owners_away and person_count > 0:
        alert_counter += 1
        if alert_counter >= CONSECUTIVE_FRAMES_THRESHOLD and not alert_sent:
            send_alert(person_count)
            alert_sent = True
    else:
        alert_counter = 0
        alert_sent = False  # room is empty again — ready to alert on the next intrusion
