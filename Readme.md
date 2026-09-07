# Edge AI — Loitering Detection on Constrained Hardware

A real-time person detection and loitering alert pipeline built to research how far commercial-grade surveillance AI can be pushed down to cheap, low-power microcontrollers — no dedicated AI chip required.

**Research angle:** Commercial systems (Hikvision AcuSense, Ambarella CV7) solve this on purpose-built AI silicon. This project documents the accuracy vs. model-size vs. latency tradeoff when you strip that away and target plain MCUs (STM32, ESP32).

---

## Project Phases

| Phase | Focus | Status |
|---|---|---|
| 1 | Environment setup | Done |
| 2 | Dataset & scope — loitering detection baseline | **In progress** |
| 3 | Fine-tune on scoped dataset, record mAP |  |
| 4 | INT8 quantization (post-training + QAT comparison) |  |
| 5 | STM32Cube.AI / Edge Impulse EON simulation |  |
| 6 | Ablation table — size / accuracy / latency |  |
| 7 | Writeup |  |

---

## What it does (Phase 2 scope)

**Loitering detection** — tracks each detected person by ID across frames and fires an alert once they have been present continuously for longer than a configurable threshold (default: 5 seconds). Uses [YOLOv8n](https://docs.ultralytics.com/) + ByteTrack, runs entirely on CPU, no GPU required.

Why loitering and not full anomaly detection: full weakly-supervised anomaly detection (UCF-Crime style) is a genuinely hard open research problem. The tractable contribution here is the compression and deployment pipeline, not inventing a new detector.

---

## Prerequisites

**Python 3.8 or higher.** Check with:
```bash
python3 --version
```

Download from [python.org](https://www.python.org/downloads/) if needed.

---

## Setup (one-time)

**1. Clone the repo**
```bash
git clone https://github.com/akash-reddy-k/edgeAi.git
cd edgeAi
```

**2. Create a virtual environment**
```bash
python3 -m venv edgeai_env
```

**3. Activate it**

macOS / Linux:
```bash
source edgeai_env/bin/activate
```
Windows:
```bash
edgeai_env\Scripts\activate
```

You'll see `(edgeai_env)` in your prompt when it's active.

**4. Install dependencies**
```bash
pip install -r requirements.txt
```

> The first run downloads YOLOv8n weights (~6 MB) automatically. One-time internet connection needed.

---

## Usage

### Loitering detection
```bash
python edgeAi.py files/TwoKids.mp4
```

Expected output:
```
Running loitering detection on: files/TwoKids.mp4  (threshold: 5.0s @ 30.0 fps)
ALERT: Person #1 loitering — present for 5.1s
```

Use your own video:
```bash
python edgeAi.py /path/to/your/video.mp4
```

Change the loitering threshold:
```bash
python edgeAi.py files/TwoKids.mp4 --loiter-seconds 10
```

### Baseline evaluation
Run this before any quantization to record the FP32 baseline. This number is the benchmark everything else is measured against.

Single video:
```bash
python eval.py files/TwoKids.mp4
```

Whole directory (e.g. ShanghaiTech dataset):
```bash
python eval.py data/shanghaitech/ --output results/baseline.json
```

Results are saved as JSON in `results/`.

---

## Dataset — ShanghaiTech Campus

Download the dataset from Kaggle and extract it into `data/shanghaitech/`:

https://www.kaggle.com/datasets/nikanvasei/shanghaitech-campus-dataset-test

Then run the baseline eval:
```bash
python eval.py data/shanghaitech/ --output results/baseline.json
```

---

## Project structure

```
edgeAi/
├── edgeAi.py              # loitering detection (main script)
├── eval.py                # baseline detection evaluation
├── download_dataset.py    # ShanghaiTech download helper
├── requirements.txt
├── models/                # model weights — gitignored, auto-downloaded
│   └── yolov8n.pt
├── data/                  # datasets — gitignored
│   └── shanghaitech/
├── results/               # eval output JSON — gitignored
└── files/                 # sample test videos
    ├── TwoKids.mp4
    ├── SumithSleeping.mp4
    └── concert.mp4
```

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'ultralytics'`**
Virtual environment isn't active. Run `source edgeai_env/bin/activate` (macOS/Linux) or `edgeai_env\Scripts\activate` (Windows).

**`No such file or directory`**
Wrong video path, or you're not running from inside the `edgeAi` folder.

**Script finishes with "No loitering detected"**
Nobody stayed in frame long enough to hit the threshold. Try `--loiter-seconds 2` for a quick test, or use a video with people standing around.

**Kaggle authentication failed**
Make sure `kaggle.json` is in the right place and you've run `chmod 600 ~/.kaggle/kaggle.json`.
