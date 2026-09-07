# Edge AI — Local Person Detection & Alert System

Detects people in a video file and prints an alert if someone is spotted while the house owners are away. Uses [YOLOv8](https://docs.ultralytics.com/) — a lightweight AI model that runs entirely on your laptop, no internet or GPU required after setup.

---

## What it does

- Reads any video file you give it (MP4, AVI, MOV, etc.)
- Scans every 5th frame for people
- Once it sees a person across 5 consecutive sampled frames, it prints an alert
- Resets and watches again if the area becomes empty

---

## Prerequisites

You only need **Python 3.8 or higher** installed on your machine.

**Check if you have it:**
```bash
python3 --version
```

If you see `Python 3.x.x`, you're good. If not, download it from [python.org](https://www.python.org/downloads/).

---

## Setup (one-time)

**1. Download the project**

```bash
git clone https://github.com/akash-reddy-k/edgeAi.git
cd edgeAi
```

**2. Create a virtual environment**

This keeps the project's packages separate from the rest of your system.

```bash
python3 -m venv edgeai_env
```

**3. Activate the virtual environment**

On macOS / Linux:
```bash
source edgeai_env/bin/activate
```

On Windows:
```bash
edgeai_env\Scripts\activate
```

You should see `(edgeai_env)` appear at the start of your terminal prompt.

**4. Install dependencies**

```bash
pip install -r requirements.txt
```

This installs `ultralytics` and everything it needs (PyTorch, OpenCV, etc.). It may take a minute.

---

## Running the script

```bash
python edgeAi.py files/TwoKids.mp4
```

Replace `files/TwoKids.mp4` with the path to any video you want to analyse.

**What you'll see:**

```
ALERT: 2 person(s) detected while house is away!
```

If no people are detected, the script finishes silently.

> **Note:** The first time you run it, the YOLOv8 model weights (~6 MB) are downloaded automatically. This requires an internet connection once.

---

## Using your own video

Put any `.mp4`, `.avi`, or `.mov` file anywhere on your computer and pass its path:

```bash
python edgeAi.py /path/to/your/video.mp4
```

---

## Adjusting behaviour

Open `edgeAi.py` in any text editor to change these two settings near the top of the file:

| Setting | Default | What it does |
|---|---|---|
| `--owners-away` flag | `True` | Pass this flag to enable alert mode. Remove it to disable alerts. |
| `CONSECUTIVE_FRAMES_THRESHOLD` | `5` | How many consecutive detections before an alert fires. Higher = fewer false positives. |

Example — disable alerts (owners are home):
```bash
python edgeAi.py files/TwoKids.mp4  # --owners-away is True by default
```

---

## Troubleshooting

**`python3: command not found`**
Install Python from [python.org](https://www.python.org/downloads/) and re-open your terminal.

**`ModuleNotFoundError: No module named 'ultralytics'`**
You forgot to activate the virtual environment. Run:
```bash
source edgeai_env/bin/activate   # macOS/Linux
edgeai_env\Scripts\activate      # Windows
```
Then try again.

**`No such file or directory: 'files/TwoKids.mp4'`**
The video path is wrong. Double-check the filename and that you are running the command from inside the `edgeAi` folder.

**Script runs but prints nothing**
No people were detected in your video, or fewer than 5 consecutive sampled frames had a person. Try a video with visible people in frame for a few seconds.
