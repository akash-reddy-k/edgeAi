# One-page research statement — source

This is the markdown source for `one_pager.tex` / `one_pager.pdf` (same
directory). Edit here, then rebuild the PDF:

```bash
tectonic outreach/one_pager.tex
```

Keep this in sync with `writeup/paper.tex` by hand — it is a condensed,
email-attachable version of the same paper, not an independent document.
If a number here disagrees with the paper, the paper is the source of truth.

---

**Akash Reddy K** — Independent research / portfolio project
Code, data, full paper: https://github.com/akash-reddy-k/edgeAi

## Fitting person detection on a plain microcontroller

Commercial CCTV analytics (Hikvision AcuSense, Ambarella CV7) run person
detection on purpose-built AI silicon with dedicated neural accelerators.
This project asks how far equivalent functionality — loitering detection on
a fixed-mount camera — can be pushed onto a **plain microcontroller with no
such accelerator**, and measures the answer rather than assuming it.

**Method.** Fine-tune YOLOv8n for single-class person detection
(ShanghaiTech Campus dataset), track with ByteTrack, export to ONNX, apply
INT8 post-training quantization. An ONNX-graph profiler then computes flash
and activation-RAM footprint directly — no vendor toolchain or account
required, fully reproducible from committed code.

**Finding 1 — quantization alone is not enough.** The INT8 model is 72%
smaller than its FP32 export for a 0.2% accuracy cost, but at the stock
YOLOv8n architecture, **nothing fits any plain STM32, at any input
resolution** — flash alone overshoots the largest plain part (STM32H743,
2 MB) by 50%.

**Finding 2 — that's an architecture fact, not a hardware fact.** Profiling
shows YOLOv8n's large-object detection pathway (P5/32) accounts for 51.8% of
parameters and is functionally unused in fixed-mount overhead CCTV, where a
person never fills half the frame. Removing it cuts flash 2.8× — more than
quantization achieved on its own.

**Finding 3 — the fix doesn't cost accuracy, it improves it.** Reinvesting
that freed budget as extra width (rather than shrinking every scale
uniformly) wins on every metric in a controlled, random-init-matched
comparison, including +3.3 points of mAP50-95. The resulting architecture
fits an STM32H743 at 320×320 with roughly 50% headroom on both flash and
RAM — no NPU, no external memory.

**Honest caveat.** The accuracy numbers above are a relative comparison on a
held-constant, self-referential label set (no ground-truth bounding-box
annotations exist for this dataset); the MCU fit numbers are an analytical
lower bound, not yet verified on physical hardware. Both limitations are
stated explicitly in the full paper, and both are active next steps —
see `outreach/mcu_validation_plan.md` and `label_gt_subset.py` in the repo.

**Why this, not a bigger model.** The research contribution is the
compression-and-deployment pipeline and its diagnostic method — distinguish
what a flash budget is actually spent on before assuming quantization is the
only lever — not a new detector.

Full paper (12 pages, arXiv-style): `writeup/paper.pdf` in the repo above.
