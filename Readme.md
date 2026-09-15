# Edge AI — Loitering Detection on Constrained Hardware

A real-time person detection and loitering alert pipeline built to research how far commercial-grade surveillance AI can be pushed down to cheap, low-power microcontrollers — no dedicated AI chip required.

**Research angle:** Commercial systems (Hikvision AcuSense, Ambarella CV7) solve this on purpose-built AI silicon. This project documents the accuracy vs. model-size vs. latency tradeoff when you strip that away and target plain MCUs (STM32, ESP32).

---

## Project Phases

| Phase | Focus | Status |
|---|---|---|
| 1 | Environment setup | Done |
| 2 | Dataset & scope — loitering detection baseline | Done |
| 3 | Fine-tune on scoped dataset, record mAP | Done |
| 4 | INT8 quantization — size / accuracy / latency tradeoff | Done |
| 5 | STM32Cube.AI / Edge Impulse EON simulation | Next |
| 6 | Ablation table — size / accuracy / latency |  |
| 7 | Writeup |  |

---

## What it does

**Loitering detection** — tracks each detected person by ID across frames and fires an alert once they have been present continuously beyond a configurable time threshold (default: 5 seconds). Uses [YOLOv8n](https://docs.ultralytics.com/) + ByteTrack.

Full weakly-supervised anomaly detection is deliberately out of scope — the research contribution is the compression and deployment pipeline, not a new detector.

---

## Current results (Phase 4 — compression tradeoff)

All three variants evaluated on the **same** 4,665-image validation split with the
**same** labels, timed on the same CPU (Apple M4 Pro), `batch=1`, 640×640:

| Variant | Size | mAP50 | mAP50-95 | Precision | Recall | Inference (median ± σ) | End-to-end |
|---|---|---|---|---|---|---|---|
| PyTorch FP32 | 5.94 MB | 0.9380 | 0.7712 | 0.9278 | 0.8831 | 21.4 ± 0.8 ms | 41.0 fps |
| ONNX FP32 | 11.70 MB | 0.9380 | 0.7712 | 0.9278 | 0.8831 | 11.6 ± 2.8 ms | 57.0 fps |
| **ONNX INT8** | **3.23 MB** | **0.9361** | **0.7377** | 0.9256 | 0.8841 | 12.0 ± 0.2 ms | **65.0 fps** |

**Headline: INT8 is 72.4% smaller than ONNX FP32 for a 0.2% mAP50 cost.**

Raw data: [`results/phase4_benchmark.json`](results/phase4_benchmark.json). Latency is a
200-image sample after 20 warmup passes; accuracy is the full 4,665-image val split.

> **Read [Methodology & Limitations](#methodology--limitations) before citing these figures** —
> the absolute mAP is inflated by the labelling strategy. The *deltas between rows* are the valid result.

### What the numbers actually say

**1. ONNX export is numerically lossless.** PyTorch FP32 and ONNX FP32 agree to four
decimal places on every metric. That is a pipeline correctness check, not a coincidence —
it confirms the export stage introduces no error, so any drift in the INT8 row is
attributable to quantization alone.

**2. Quantization costs localization, not detection.** mAP50 falls 0.2% (0.9380 → 0.9361)
while mAP50-95 falls 4.3% (0.7712 → 0.7377). Recall actually rises slightly. The model still
*finds* people just as reliably; its boxes are just looser, so it loses points at strict IoU
thresholds. For loitering detection — which asks "is a person present and trackable?", not
"is this box within 90% IoU?" — the damage lands almost entirely on the axis this
application is least sensitive to.

**3. On a general-purpose CPU, INT8 buys no median speedup at all.** Across three
independent timing runs, ONNX FP32 medians were 12.25 / 12.37 / 11.57 ms and ONNX INT8
12.36 / 11.37 / 12.00 ms — fully overlapping, with INT8 *slower* in one run. The difference
is smaller than the run-to-run noise, so the honest statement is that the two are
indistinguishable in peak inference, not that INT8 wins. ONNX Runtime's CPU provider has no
fully optimized INT8 kernel path here, and the inserted quantize/dequantize ops consume the
theoretical gain. **This is the expected result, not a failure** — INT8's speed advantage
requires hardware with dedicated integer paths, which is exactly what Phase 5 measures on
STM32. Phase 4's job was to establish that the compression is accuracy-safe; Phase 5 is
where it should become fast.

**4. The real latency win is consistency, not peak speed.** INT8's standard deviation was
0.44 / 0.29 / 0.23 ms against ONNX FP32's 2.77 / 3.41 / 2.80 ms — roughly 10× tighter in
every run. The smaller weight set has a friendlier cache footprint and is less exposed to
memory-bandwidth contention. Because FP32 suffers occasional slow-frame spikes that INT8
does not, INT8 wins **end-to-end throughput in all three runs (+20%, +27%, +14%)** despite
identical median inference. For real-time surveillance this is the more useful property:
jitter is what drops frames, not average latency.

**5. The free win was ONNX export itself.** PyTorch 21.4 ms → ONNX FP32 11.6 ms is a 1.85×
CPU speedup at *identical* accuracy, before any quantization. On CPU-bound deployments that
is a larger and more reliable gain than INT8 delivers.

### Why the FP32 baseline reads 0.938 here but 0.945 in Phase 3

Phase 3 reported 0.945 from the training run, which uses ultralytics' default *rectangular*
batching. The exported ONNX graphs have a **static 640×640 input**, so they cannot consume
rectangular batches. Reusing the Phase 3 figure would have compared a rect-batched PyTorch
model against square-letterboxed ONNX models and silently attributed the difference to
quantization. `benchmark.py` therefore re-measures all three arms under identical conditions
(`rect=False`, `batch=1`). The 0.945 → 0.938 shift is the cost of that fairness control,
not model drift.

### QAT was not pursued

The Phase 4 plan called for quantization-aware training *only if* post-training
quantization degraded accuracy badly. A 0.2% mAP50 drop does not meet that bar — QAT would
add a full retraining cycle to recover a fraction of a percent on a metric whose absolute
value is already self-referential. Skipped as a deliberate scope decision.

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

## Dataset — ShanghaiTech Campus

Download from Kaggle and extract into `data/SHANGHAI_Test/`:

https://www.kaggle.com/datasets/nikanvasei/shanghaitech-campus-dataset-test

Expected layout:
```
data/SHANGHAI_Test/
├── frames/<scene_id>/*.jpg
└── SHANGHAI_test.txt
```

---

## Usage

### Loitering detection
```bash
python edgeAi.py files/TwoKids.mp4
```

Output:
```
Running loitering detection on: files/TwoKids.mp4  (threshold: 5.0s @ 30.0 fps)
ALERT: Person #1 loitering — present for 5.1s
```

Adjust the threshold:
```bash
python edgeAi.py files/TwoKids.mp4 --loiter-seconds 2    # quick test
python edgeAi.py files/TwoKids.mp4 --loiter-seconds 15   # fewer false positives
```

### Detection statistics on the dataset
Reports detection rate split by anomaly vs normal frames, using the dataset's ground-truth frame labels:
```bash
python eval.py data/SHANGHAI_Test/ --output results/baseline.json
```

### Reproducing Phase 3 training
```bash
python prepare_dataset.py    # generate labels + train/val split (~15k frames)
python train.py              # fine-tune 20 epochs, saves models/yolov8n_finetuned.pt
```

Options:
```bash
python prepare_dataset.py --conf 0.3       # stricter label generation
python train.py --epochs 30 --batch 32
```

### Reproducing Phase 4 quantization
```bash
python quantize.py     # export ONNX FP32 + ONNX INT8  (~40s)
python benchmark.py    # size / accuracy / latency across all 3 variants (~10 min, CPU)
```

`benchmark.py` writes `results/phase4_benchmark.json` and prints the tradeoff table.

Options:
```bash
python quantize.py --calib-size 2000            # larger INT8 calibration set
python benchmark.py --skip-accuracy             # latency + size only (seconds)
python benchmark.py --latency-images 200        # tighter timing sample
```

---

## Methodology & Limitations

**Honest accounting of how the Phase 3 numbers were produced, and what they do and do not prove.**

### Labelling strategy

The ShanghaiTech Campus test set ships with **frame-level anomaly labels only** — each clip is annotated with the frame range in which an anomalous event occurs. It contains no bounding-box annotations, which object detector fine-tuning and mAP computation both require.

Rather than hand-annotate 15,306 frames, labels were generated by running the COCO-pretrained YOLOv8n over every frame and persisting its person detections as YOLO-format boxes (`prepare_dataset.py`). The model was then fine-tuned on those labels.

### What this means for the reported mAP

The validation labels were produced by the same base model that was fine-tuned. The reported **mAP50 of ~0.94 therefore measures how faithfully the fine-tuned model reproduces the base model's own predictions on campus footage — not how accurately it detects people.** The absolute figure is inflated and should not be cited as a detection-accuracy result.

### Why the baseline is still usable

Phase 4 compared FP32 against INT8 on the *same* validation set with the *same* labels, in the same process, on the same hardware. Because the label set is held constant across all three arms, the **relative** degradation is a valid measurement even though the absolute anchor is not. The −0.2% mAP50 / −4.3% mAP50-95 quantization cost reported above is therefore a real result; the 0.9380 it starts from is not. The accuracy-vs-size-vs-latency tradeoff curve — the actual contribution of this project — is unaffected.

### Accepted limitation

No true-ground-truth subset was annotated. This was a deliberate scope decision: hand-labelling would strengthen the absolute claim but does not change the tradeoff curve, which is what the research question turns on. The limitation is stated rather than hidden, and the Phase 7 writeup reports the mAP figures as relative-only.

**If absolute accuracy becomes load-bearing later**, the cheapest fix is hand-annotating ~200 validation frames to anchor the relative numbers to one honest absolute measurement.

### INT8 calibration leakage (Phase 4)

INT8 quantization needs representative images to calibrate activation ranges. Ultralytics
draws these from the `val:` split of whatever dataset yaml it is handed — so passing the
project's `dataset.yaml` would have calibrated the INT8 model on **the same images used to
evaluate it**, inflating its apparent accuracy and corrupting the FP32-vs-INT8 delta.

`quantize.py` instead builds a dedicated calibration set by subsampling 1,000 images from
the **train** split and writes a separate `calibration.yaml` whose `val:` points at it.
Calibration and evaluation therefore use strictly disjoint data. The subsample is evenly
spaced across the split rather than the first N images, so all 20 training scenes are
represented rather than only the alphabetically-first few.

### Benchmark fairness controls

- **`batch=1` and `rect=False` on all three arms.** The ONNX graphs are exported with a
  static 640×640 input, so ultralytics' default rectangular batching would have fed the
  PyTorch arm different input shapes than the ONNX arms — and that difference would have
  been misread as a quantization effect. See "Why the FP32 baseline reads 0.938 here".
- **Latency measured on CPU, not MPS.** The Phase 5 deployment target is an MCU with no AI
  accelerator, so MPS numbers would flatter the model in a way that does not transfer.
- **Timing sample is evenly spaced across the val split**, with warmup passes discarded, and
  reported as a median — CPU scheduling spikes skew the mean.
- **Thermal throttling was checked, not assumed.** Arms run sequentially, so a heat-related
  slowdown would penalise whichever runs last — and INT8 runs last. Throttling would show up
  as drifting, inflated timings, but INT8 recorded the *lowest* variance of any arm in every
  run (σ ≈ 0.2–0.4 ms) with a median matching FP32, which rules out an ordering artifact.
  `pmset -g therm` reported no thermal or performance warning during the runs. The higher
  variance on the other arms tracks external CPU contention, not temperature.
- **Latency was measured three times, not once.** The first two runs disagreed on INT8 by
  ~8% — larger than the FP32-vs-INT8 gap itself. That is why this README reports the two arms
  as indistinguishable in median inference rather than claiming a speedup a single run would
  have appeared to support.

### Other notes

- Train/val split is **scene-level**, not frame-level — no frames from a validation scene appear in training, so there is no temporal leakage between near-identical adjacent frames.
- Val split holds 8 scenes (4 anomaly / 4 normal) against 20 training scenes.
- Fine-tuning ran on Apple MPS; results are not bit-reproducible across backends as some MPS ops lack deterministic implementations.
- Phase 4 latency was measured on an Apple M4 Pro (12 cores). Absolute milliseconds are machine-specific; the ratios between variants are the portable result.

---

## Project structure

```
edgeAi/
├── edgeAi.py              # loitering detection (main script)
├── eval.py                # detection stats vs ground-truth frame labels
├── prepare_dataset.py     # pseudo-label generation + train/val split
├── train.py               # fine-tuning
├── quantize.py            # ONNX FP32 + INT8 export w/ leak-free calibration
├── benchmark.py           # size / accuracy / latency tradeoff table
├── requirements.txt
├── models/                # weights — gitignored
│   ├── yolov8n.pt                    # COCO-pretrained base
│   ├── yolov8n_finetuned.pt          # Phase 3 output
│   ├── yolov8n_finetuned_fp32.onnx   # Phase 4 — ONNX FP32
│   └── yolov8n_finetuned_int8.onnx   # Phase 4 — ONNX INT8
├── data/                  # datasets — gitignored
│   ├── SHANGHAI_Test/         # raw dataset
│   └── yolo_dataset/          # generated labels + symlinked splits
│       └── calibration.yaml       # INT8 calibration split (from TRAIN)
├── results/               # eval + benchmark JSON
├── runs/                  # ultralytics training output — gitignored
└── files/                 # sample test videos
```

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'ultralytics'`**
Virtual environment isn't active. Run `source edgeai_env/bin/activate` (macOS/Linux) or `edgeai_env\Scripts\activate` (Windows).

**`No such file or directory`**
Wrong path, or you're not running from inside the `edgeAi` folder.

**Script finishes with "No loitering detected"**
Nobody stayed in frame long enough to hit the threshold. Try `--loiter-seconds 2`.

**`Dataset not found` when running train.py**
Run `python prepare_dataset.py` first — training needs the generated label set.

**Training is very slow**
`train.py` targets Apple MPS. On a non-Apple machine, change `device="mps"` to `device=0` (CUDA) or `device="cpu"` in `train.py`.
