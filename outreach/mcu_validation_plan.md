# MCU hardware validation — runbook

**Status: not yet executed. Needs a physical board.** Everything in Phase 5/6
(`results/phase5_mcu_profile.json`, `results/phase6_ablation.json`) is an
*analytical* claim — flash and activation RAM computed directly from ONNX
graph shapes, deliberately without STM32Cube.AI or Edge Impulse (both need
vendor accounts, see README "Method and its limits" under Phase 5 and 6).
That was the right call for a fast, reproducible, no-account-needed feasibility
sweep across six architectures and six resolutions. It is not the same as
proof on real silicon. This document is the exact sequence to run once a
board is in hand, so that session is execution, not re-deriving a plan.

**Target claim to verify:** P3/P4 w0.25 at 320×320 fits an STM32H743 —
992.4 KB flash against a 2,048 KB budget, 732.0 KB RAM against 1,024 KB
(`results/phase6_ablation.json`, envelope section). Both roughly half-budget.

---

## 1. Hardware

**Buy: NUCLEO-H743ZI2** (~$25–30, ST's official distributors or Mouser/Digi-Key).
Reasons over the alternative (STM32H743I-DISCO, ~$80):
- Same MCU (STM32H743ZI, 2 MB flash / 1 MB RAM) — matches the exact budget row
  this project already analyzed. No need to re-derive a different part's limits.
- The Discovery board's extra peripherals (display, audio codec) are not used
  by this validation — a Nucleo is strictly cheaper for the same silicon.
- ST-LINK/V3 debugger is onboard — no separate programmer needed.

No camera module needed for this validation pass. The question being answered
is "does the model fit and how fast does it run," not "does the live pipeline
work end-to-end" — feed it static test images from the existing val set, the
same thing `benchmark.py` already does on CPU.

## 2. Toolchain

**Install STM32CubeIDE** (free, ST account required — this is a vendor account,
unlike the Python stack used so far, and it is specifically why this step was
deferred out of Phase 5/6). Then add the **X-CUBE-AI** expansion pack via
STM32CubeMX (also free). X-CUBE-AI:
- Ingests an ONNX graph directly.
- Generates C inference code using ST's own optimized kernels.
- Reports flash and RAM usage for the *generated project*, which is the
  vendor-measured counterpart to this project's analytical numbers — the
  natural cross-check.
- Has a "Validate on target" mode: flashes a thin test harness, feeds it
  input vectors, and reports per-layer and total inference cycle counts
  measured on the actual core. This is the real latency number Phase 4/5/6
  never had access to.

## 3. Get the right ONNX artifact

**Gotcha:** `models/ablation_cache/p34_int8.onnx` (the committed cache) was
exported at `--imgsz 640` (ablate.py's default), and `export_int8()` in
`ablate.py` caches by variant name only, not by resolution — so it will NOT
reuse or coexist with a 320×320 version under the same filename.

```bash
# back up the existing 640px cache before overwriting it
cp models/ablation_cache/p34_int8.onnx models/ablation_cache/p34_int8_640.onnx.bak

# force a fresh export with a 320x320 INPUT (not just a 320px arena measurement)
python ablate.py --variants p34 --imgsz 320 --skip-envelope
```

This produces `models/ablation_cache/p34_int8.onnx` built for a 320×320 input
— the artifact to hand to X-CUBE-AI. Confirm it's the right one before
importing: `python -c "import onnx; m=onnx.load('models/ablation_cache/p34_int8.onnx'); print(m.graph.input[0].type.tensor_type.shape)"`
should show 320 in the spatial dims.

## 4. Import and generate

In STM32CubeMX: new project → NUCLEO-H743ZI2 → Software Packs → X-CUBE-AI →
Add network → point at the `p34_int8.onnx` from step 3 → Analyze.

Record, before generating code:
- Reported flash footprint (compare directly to the 992.4 KB analytical figure).
- Reported RAM / activation buffer size (compare to the 732.0 KB figure).

Expect the vendor numbers to be *somewhat higher* — this project's analytical
model assumes a perfect memory planner and excludes runtime/application code
(stated explicitly in both Phase 5 and 6 "Method and its limits" sections).
If vendor flash/RAM come in lower than the analytical figures, that is a bug
in the analytical model worth tracking down, not a result to just accept.

## 5. Validate on target, record latency

Generate code → build → flash → X-CUBE-AI's "Validate on target" against a
handful of real val-split images (reuse `data/yolo_dataset/images/val/`, same
ones `benchmark.py` already times on CPU — comparable inputs, comparable
story). Record:
- Per-inference latency (cycles and ms, at the H743's clock speed).
- Whether the on-device output (boxes/confidences) roughly matches the ONNX
  Runtime output for the same image — a basic correctness check, not just a
  size/speed one.

## 6. Write up the result

Add a short section to `results/` (e.g. `results/phase8_mcu_validation.json`)
and a paragraph to the paper/README stating measured vs. analytical flash,
RAM, and the new on-device latency number — explicitly labeled as vendor-
measured, distinct from the Phase 5/6 analytical figures, so the two don't
get conflated. This is the single change that would most strengthen the
paper's central hardware-feasibility claim for a technical reader who knows
to ask "did you actually run this, or just calculate it?"

## Fallback if a board isn't available soon

Edge Impulse's free tier can take an ONNX/TFLite model and report an
estimated flash/RAM/latency profile for a named target MCU **without owning
the hardware** (it uses characterized device profiles). It's a weaker claim
than an actual on-device run, but still an independent, vendor-side
cross-check of the analytical numbers, and it needs only a free account, not
a purchase. Worth doing even before the board arrives.
