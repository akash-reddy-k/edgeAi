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
| 5 | MCU deployment feasibility — flash / RAM profiling | Done |
| 6a | Architecture ablation — flash / RAM cost, no training | Done |
| 6b | Train surviving variants — accuracy cost | Next |
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

## Phase 5 — Does it actually fit on a microcontroller?

**Short answer: no. Not on any plain STM32, at any input resolution.** This is the
project's central negative result, and it is a more useful finding than a success would
have been.

Two numbers decide MCU deployability: **flash** (weights, stored once) and **activation
RAM** (the working arena, reused across layers). `mcu_profile.py` computes both directly
from the ONNX graph — reproducible from this repo, no vendor account required.

### Flash — 3,117 KB, essentially fixed regardless of input resolution

| Component | Size | Scales with |
|---|---|---|
| INT8 weights | 2,930.3 KB | architecture |
| Anchor / stride constants (FP32) | 164.1 KB | **input resolution** |
| INT32 biases | 21.3 KB | architecture |
| Quantization scales (FP32) | ~1.0 KB | tensor count |
| Misc (INT64) | 0.2 KB | — |
| **Total** | **3,117 KB (3.04 MB)** | |

Sanity check: 3,005,843 parameters × 1 byte = 2,935 KB theoretical against 2,930.3 KB
measured. The accounting matches to 0.2%.

**Correction (found in Phase 6).** An earlier version of this section listed the 165.1 KB
of FP32 as "scales / zero-points" and claimed that shrinking the input "does not remove a
single byte of flash." Both were wrong. That block is almost entirely the Detect head's
baked-in anchor and stride constants — three tensors over all 8,400 anchor points, 20 bytes
per point — and those *do* scale with resolution. Actual quantization scales are ~1.0 KB,
because ultralytics quantizes per-tensor, not per-channel.

So the honest version: **2,952.8 KB is resolution-independent, 164.1 KB is not.** Dropping
640×640 → 96×96 frees 161.9 KB, taking flash to 2,955 KB. Against the 2,048 KB of the
largest plain STM32 (H743) the model still overshoots by 44%, so the conclusion is
unchanged — but "not a single byte" was an overstatement, and on the sub-megabyte models of
Phase 6 that anchor block grows to ~15% of the flash bill, where it stops being a rounding
error.

### Activation RAM — scales with resolution, bottlenecked at layer one

| Input | Arena (SiLU fused) | Arena (unfused) |
|---|---|---|
| 640×640 | 2,800 KB | 4,800 KB |
| 320×320 | 700 KB | 1,200 KB |
| 256×256 | 448 KB | 768 KB |
| 192×192 | 252 KB | 432 KB |
| 128×128 | 112 KB | 192 KB |
| 96×96 | 63 KB | 108 KB |

The peak sits at **node 0** — the very first convolution — at every resolution, because
that is where the tensors are largest, before any downsampling. The 640×640 figure is
hand-verifiable: input (3×640×640 = 1,200 KB) + first conv output (16×320×320 = 1,600 KB)
= 2,800 KB.

### Feasibility frontier

| Target | Flash | RAM | Max workable input |
|---|---|---|---|
| STM32F746 (M7) | 1 MB | 320 KB | **none** |
| STM32F767 (M7) | 2 MB | 512 KB | **none** |
| STM32H743 (M7 480 MHz) | 2 MB | 1 MB | **none** |
| ESP32-S3 | 8 MB ext | 512 KB | 256×256 |
| ESP32-S3 + 8 MB PSRAM | 8 MB ext | 8 MB | 640×640 |
| STM32N657 (M55 + NPU) | 8 MB ext | 4.2 MB | 640×640 |

**Reading this table honestly:** the only targets that fit are the ones that cheat. The
ESP32-S3 relies on *external* flash and, past 256×256, external PSRAM — both off-chip, both
far slower than internal SRAM. The STM32N657 fits comfortably, but it carries a Neural-ART
NPU, which is precisely the dedicated AI silicon this project set out to do without. It
proves the thesis by contradiction.

The genuinely useful conclusion: **for this architecture, flash is the binding constraint,
not RAM, and not latency.** Quantization already bought 72%; reaching a plain STM32H743
needs roughly another 2× off the *weights*, which INT8 cannot give. That requires
architectural change — which is what Phase 6 tests.

> **Phase 6 update:** that framing holds only while flash is so far over budget that RAM
> never gets to matter. Once an architectural change fixes flash, RAM becomes binding —
> and no architectural change touches it, because the arena peak is the input buffer plus
> the first convolution. The two constraints have different causes and need different
> levers. See [Phase 6](#phase-6--architecture-ablation-what-actually-deploys).

### Method and its limits

- **This is an analytical model, not a vendor measurement.** STM32Cube.AI and Edge Impulse
  EON were not run — both require vendor accounts, and neither is installed here. Their
  numbers would be somewhat *higher* (runtime overhead, buffer alignment, imperfect
  planning), which strengthens rather than weakens the negative conclusion.
- **The arena is a lower bound.** It assumes a perfect memory planner that frees every
  buffer the instant it dies. Real runtimes do worse. A lower bound is the conservative
  choice for a feasibility claim: if even the floor does not fit, nothing does.
- **SiLU fusion is modelled explicitly.** ONNX stores SiLU as `Mul(y, Sigmoid(y))`, leaving
  three full-size tensors live at once. Every real MCU runtime emits a single fused kernel.
  Skipping this pass overstates the arena by up to 1.7×, and since YOLOv8's peak sits on the
  first SiLU, that error would land straight on the headline. Both figures are reported above.
- Flash excludes the inference runtime and application code, so the real requirement is
  higher still.

---

## Phase 6 — Architecture ablation: what actually deploys

Phase 5 ended on a negative: nothing fits a plain STM32, at any resolution. Phase 6 asks
whether that is a fact about *microcontrollers* or a fact about *YOLOv8n*. It turns out to
be the latter.

**Short answer: a P5-free YOLOv8 at 320×320 fits an STM32H743 with >50% headroom on both
flash and RAM** — no NPU, no external memory. `ablate.py` measures it.

### These numbers need no training

Flash footprint and activation arena are fixed by the *architecture* — the shapes of the
weight tensors and of the intermediates. They do not depend on the values the weights take,
so an untrained model quantizes to exactly the same byte count as a trained one. The whole
sweep below runs without a single training step, which splits Phase 6 cleanly:

| Stage | Question | Cost |
|---|---|---|
| 1 — `ablate.py` | what does each architecture **cost**? | no training, exact |
| 2 — `train.py` | what does each architecture **buy**? | training required |

Only variants that survive Stage 1 are worth spending Stage 2 time on. **Stage 2 is not yet
run, so no accuracy claim is made here.**

### Where the parameters actually are

Profiling stock YOLOv8n (nc=1) by layer shows the weights are not spread evenly:

| Region | Params | Share |
|---|---|---|
| P5/32 pathway (backbone 7/8/9 + head 19/21) | 1,561,088 | **51.8%** |
| Detect head | 751,507 | 25.0% |
| P3+P4 feature trunk | 698,448 | 23.2% |

**Over half the network exists to detect large objects.** P5/32 fires on objects roughly
≥320 px in a 640 px frame — a person filling half the frame height. In fixed-mount overhead
CCTV that essentially never happens, so on this domain that half of the model is capacity
the deployment never uses. `configs/yolov8-p34.yaml` removes it: SPPF moves onto P4, the
FPN/PAN becomes two-scale, and Detect regresses at strides [8, 16].

### Cost of each architecture (640×640, measured)

| Variant | Params | Flash | vs base | Arena |
|---|---|---|---|---|
| P3/P4/P5 w0.25 *(baseline)* | 3,011,043 | 3,116.9 KB | 1.00× | 2,800 KB |
| P3/P4/P5 w0.1875 | 1,799,187 | 1,931.1 KB | 0.62× | 2,800 KB |
| P3/P4/P5 w0.125 | 912,731 | 1,063.0 KB | 0.34× | 2,000 KB |
| **P3/P4 w0.25** | 969,698 | **1,109.5 KB** | **0.36×** | 2,800 KB |
| P3/P4 w0.1875 | 604,226 | 751.6 KB | 0.24× | 2,800 KB |
| P3/P4 w0.125 | 333,242 | 485.9 KB | 0.16× | 2,000 KB |

Dropping P5 alone gives **2.8× off flash** — more than INT8 quantization did, and it clears
the 2,048 KB barrier that Phase 5 identified as unreachable.

### The finding that changes the picture

Look at the arena column: it is almost *independent of architecture*. The peak sits at node
0 in every variant, and consists of the input buffer (3×640×640 = 1,200 KB) plus the first
convolution's output. Everything downstream is smaller. So:

> **Flash is set by architecture. RAM is set by input resolution.**
> They are orthogonal, and neither lever alone is sufficient.

This corrects Phase 5's "flash is the binding constraint, not RAM." That was true only while
flash was so far over budget that RAM never got to matter. Fix flash architecturally and RAM
immediately becomes binding — and no architectural change touches it, because you cannot
shrink the input image by rearranging the network. The real deployment question is
two-dimensional.

### Deployment envelope — what actually fits

Largest input resolution that fits, on plain (no-NPU, no-PSRAM) parts:

| Target | Flash / RAM | Baseline | P3/P4 w0.25 | P3/P4 w0.125 |
|---|---|---|---|---|
| STM32F746 (M7 216 MHz) | 1 MB / 320 KB | none | **192×192** | 192×192 |
| STM32F767 (M7 216 MHz) | 2 MB / 512 KB | none | **256×256** | 256×256 |
| STM32H743 (M7 480 MHz) | 2 MB / 1 MB | none | **320×320** | 416×416 |

At the headline point — **P3/P4 w0.25 at 320×320 on an STM32H743** — the requirement is
992.4 KB flash against 2,048 KB, and 732.0 KB RAM against 1,024 KB. Both roughly half the
budget, leaving real room for the runtime and application code that these figures exclude.

Phase 5's row for the H743 read *"NO resolution fits."* It now reads 320×320.

### The Stage 2 experiment

Two variants land within 5.9% of the same parameter count but spend it completely
differently:

| Variant | Params | Flash | How the budget is spent |
|---|---|---|---|
| P3/P4/P5 w0.125 | 912,731 | 1,063.0 KB | all three scales, half channel width |
| P3/P4 w0.25 | 969,698 | 1,109.5 KB | two scales, full channel width |

Because flash says they cost the same, this is a fair architectural question rather than a
size comparison: **at a fixed budget, is it better to keep every scale and thin it, or to
keep full width and drop the scale the domain never uses?** Stage 2 answers it by training
both on identical data. The hypothesis is that P3/P4 w0.25 wins on this domain — but it is
a hypothesis, and the P5 cut gives up large-object detection outright, so it is a
domain-specific trade and not a general improvement to YOLOv8.

### Method and its limits

- Flash is **measured** from each variant's INT8 ONNX initializers, not estimated.
- **Arena is measured on the FP32 export, deliberately.** The INT8 graph is in QDQ form,
  which interposes Quantize/Dequantize nodes between `Conv → Sigmoid → Mul`. That defeats
  the SiLU fusion pass and counts the Q/DQ intermediates as live, inflating the arena 2.2×
  (2,800 KB → 6,135 KB) and silently flipping feasibility verdicts. Activation *shapes* are
  identical in both graphs, so FP32 is the correct thing to analyse.
  `mcu_common.profile_arena` now raises on a QDQ graph rather than returning a wrong number.
- Flash at resolutions other than 640 combines the measured weight bytes with an analytic
  anchor-constant term (20 bytes per anchor point), which the script validates against
  measurement at 640 before relying on it.
- Same lower-bound caveats as Phase 5: perfect memory planner assumed, runtime and
  application code excluded.
- **No accuracy has been measured for any variant.** Everything above is cost.

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

### Reproducing Phase 5 MCU profiling
```bash
python mcu_profile.py    # flash + activation RAM vs MCU budgets (~30s)
```

Writes `results/phase5_mcu_profile.json` and prints the feasibility frontier.

Options:
```bash
python mcu_profile.py --resolutions 640 320 160 96   # custom resolution sweep
python mcu_profile.py --runtime-overhead-kb 64       # stricter RAM reserve
```

### Reproducing Phase 6 architecture ablation
```bash
python ablate.py    # flash + RAM for 6 architectures, no training (~4 min)
```

Writes `results/phase6_ablation.json` and prints the deployment envelope. Reuses the
leak-free calibration set built by `quantize.py`, so run that first.

Options:
```bash
python ablate.py --variants baseline p34       # subset of architectures
python ablate.py --skip-envelope               # skip the resolution sweep (faster)
python ablate.py --skip-arena                  # flash only
python ablate.py --resolutions 640 320 160     # custom envelope sweep
```

INT8 and FP32 exports are cached in `models/ablation_cache/`; delete it to force a rebuild.

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
├── mcu_common.py          # shared flash + arena analysis (Phase 5 & 6)
├── mcu_profile.py         # flash + activation RAM vs real MCU budgets
├── ablate.py              # Phase 6 — architecture ablation, no training
├── configs/               # architecture variants for the ablation
│   ├── yolov8-scaled.yaml     # stock P3/P4/P5 + narrow width rungs
│   └── yolov8-p34.yaml        # P5 branch removed, two-scale head
├── requirements.txt
├── models/                # weights — gitignored
│   ├── yolov8n.pt                    # COCO-pretrained base
│   ├── yolov8n_finetuned.pt          # Phase 3 output
│   ├── yolov8n_finetuned_fp32.onnx   # Phase 4 — ONNX FP32
│   ├── yolov8n_finetuned_int8.onnx   # Phase 4 — ONNX INT8
│   ├── profile_cache/                # Phase 5 — per-resolution export cache
│   └── ablation_cache/               # Phase 6 — per-variant export cache
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
