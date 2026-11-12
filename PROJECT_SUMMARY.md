# Edge AI — Project Summary

A retrospective on what this project set out to do, what was actually built, what the
numbers say, and how well the original aim was met. Written after all 7 phases closed out
on `main`. Sources: `Readme.md`, `writeup/paper.pdf`, and the raw result files in
`results/*.json`.

---

## Aim

Commercial surveillance AI (Hikvision AcuSense, Ambarella CV7) runs loitering/anomaly
detection on purpose-built AI silicon. The question this project asked: **how far can that
same capability be pushed down onto a plain, cheap microcontroller — no dedicated NPU —
and where exactly does it break?**

Concretely: build a real-time person-detection + loitering-alert pipeline (YOLOv8n +
ByteTrack), then push it through a compression pipeline (quantization, then architecture
surgery) and measure, honestly, whether the result fits a real MCU's flash and RAM budget.
The explicit non-goal was inventing a new detector or solving general weakly-supervised
anomaly detection — the research contribution is the **compression-and-deployment
pipeline**, not the model.

---

## What was built, phase by phase

| Phase | Question asked | Answer found |
|---|---|---|
| 1 | Environment setup | — |
| 2 | Dataset & scope | Loitering detection on ShanghaiTech Campus, person-tracking via ID + dwell-time threshold |
| 3 | Fine-tune, record mAP | YOLOv8n fine-tuned on pseudo-labels; mAP50 ≈ 0.945 (rect-batched) |
| 4 | Does INT8 quantization pay for itself? | Yes on size (-72%), free on accuracy (-0.2% mAP50), a wash on raw CPU latency, a real win on jitter/throughput |
| 5 | Does the quantized model fit a real MCU? | **No — not on any plain STM32, at any resolution.** Flash is the binding constraint |
| 6a | Is that a fact about MCUs or about YOLOv8n? | About YOLOv8n — a P5-free, narrower variant fits with >50% headroom |
| 6b | Does the architecture change cost accuracy? | No — the fitting variant (P3/P4 w0.25) *beat* the alternative on every metric |
| 7 | Writeup | 12-page arXiv-style paper, `writeup/paper.pdf`, fully reproducible from committed JSON/figures |

---

## Key results

**Compression (Phase 4).** All three variants evaluated on the same 4,665-image val split:

| Variant | Size | mAP50 | mAP50-95 | Median inference | End-to-end fps |
|---|---|---|---|---|---|
| PyTorch FP32 | 5.94 MB | 0.9380 | 0.7712 | 21.4 ms | 41.0 |
| ONNX FP32 | 11.70 MB | 0.9380 | 0.7712 | 11.6 ms | 57.0 |
| **ONNX INT8** | **3.23 MB** | 0.9361 | 0.7377 | 12.0 ms | **65.0** |

INT8 is 72.4% smaller for a 0.2% mAP50 cost. The honest nuance: on a general-purpose CPU,
INT8 buys **no median speedup** — ONNX Runtime has no fully optimized INT8 kernel path
here, and FP32-vs-INT8 medians overlapped across three independent runs. What INT8 actually
wins is **consistency** (10× tighter latency variance) and therefore end-to-end throughput
(+14–27%) — jitter drops frames, not average latency, which matters more for real-time
surveillance than the median does.

**MCU feasibility, first pass (Phase 5).** Two numbers decide deployability: flash (3,117 KB)
and activation RAM (2,800 KB at 640×640). Against every plain STM32 — F746, F767, even the
H743 at 2 MB flash / 1 MB RAM — **nothing fits, at any input resolution**. Only parts that
"cheat" fit: ESP32-S3 with external PSRAM, or the STM32N657, which carries its own NPU (the
exact dedicated silicon this project set out to avoid). This negative result was treated as
the project's most useful finding, not a failure to report around.

**Architecture ablation (Phase 6).** Profiling showed the P5/32 pathway — the part of
YOLOv8n that detects large objects — accounts for **51.8% of all parameters**, and fixed
overhead CCTV essentially never has objects that large. Removing it (`configs/yolov8-p34.yaml`)
cuts flash 2.8×, more than quantization did on its own. The deeper finding: **flash is set
by architecture, RAM is set by input resolution — they're orthogonal, and no single lever
fixes both.** The resulting deployment envelope:

| Target | Baseline | P3/P4 w0.25 |
|---|---|---|
| STM32F746 (1 MB / 320 KB) | none | 192×192 |
| STM32F767 (2 MB / 512 KB) | none | 256×256 |
| STM32H743 (2 MB / 1 MB) | none | **320×320** |

Headline: P3/P4 w0.25 at 320×320 on an STM32H743 uses roughly half the flash and RAM
budget — no NPU, no external memory. Phase 5's "NO resolution fits" became "320×320 fits,
with headroom."

**Did the architecture change cost accuracy?** No — trained from random init on identical
data (ShanghaiTech, 20 epochs, patience 8), P3/P4 w0.25 beat the alternative same-budget
variant (P3/P4/P5 w0.125) on every metric, most notably +3.3 points of mAP50-95 (0.7216 vs.
0.6886). The capacity spent on large-object detection in the baseline was capacity this
domain wasn't using — reinvesting it as width on the scales that matter measurably helped.

---

## How well the aim was met

**On its own terms, yes — including by delivering a negative result honestly.** The project
didn't just ask "can we shrink the model" — it asked "does it actually fit," found that it
didn't under the first (quantization-only) strategy, and then correctly diagnosed *why*
(architecture, not hardware) and fixed it with a second strategy. That two-step structure —
disprove the naive approach, then find the lever that actually works — is a stronger
result than either phase alone.

**What keeps it honest rather than a clean win:**
- The absolute mAP (~0.94) is **self-referential** — labels came from the same base model
  being fine-tuned, since ShanghaiTech ships no bounding boxes. This is disclosed
  prominently (README's Methodology & Limitations section, and as the bolded opening line
  of the paper's Limitations section) rather than buried. The *relative* deltas between
  variants/quantization levels are the valid, load-bearing results; the absolute number is
  not.
- The MCU numbers are **analytical** (computed from ONNX graph shapes), not measured on
  vendor toolchains (STM32Cube.AI, Edge Impulse) or physical hardware. The paper argues
  this is a conservative lower bound, but it's still unverified against real silicon.
- Stage 2 training comparison is a **single seed per variant** — no variance estimate yet.
- QAT was deliberately skipped (0.2% mAP50 drop didn't meet the bar that would justify it) —
  a scope decision, not an oversight.

**Net assessment:** the project answers its own research question with evidence, states
clearly which numbers are solid (compression deltas, architecture-driven flash/RAM
footprints) versus which are provisional (absolute accuracy, unmeasured hardware latency),
and ends with a concrete, reproducible artifact (`writeup/paper.pdf`) rather than just a
README. The biggest open gap for anyone wanting to cite the absolute accuracy number is the
missing hand-annotated ground truth; the biggest open gap for anyone wanting to cite the
hardware claim is the missing vendor/physical measurement. Both are explicitly flagged as
future work rather than glossed over.

---

## Reproducibility chain

Every number above traces back to a committed artifact — no number in this summary, the
README, or the paper was generated specifically for documentation:

```
prepare_dataset.py → train.py            (Phase 3: results live in models/, runs/)
quantize.py → benchmark.py               (Phase 4: results/phase4_benchmark.json)
mcu_profile.py                           (Phase 5: results/phase5_mcu_profile.json)
ablate.py → train.py --variant {w125,p34} (Phase 6: results/phase6_ablation.json)
writeup/make_figures.py → paper.tex → tectonic → paper.pdf   (Phase 7)
```

---

*Generated as a project retrospective. See [`Readme.md`](Readme.md) for full usage/setup
instructions and [`writeup/paper.pdf`](writeup/paper.pdf) for the formal writeup.*
