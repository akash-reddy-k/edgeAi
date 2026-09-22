"""
Phase 6 — Architecture ablation for the MCU flash budget.

Phase 5 ended on a hard negative: the INT8 model needs ~3,117 KB of flash while
the largest plain STM32 (H743) offers 2,048 KB. INT8 cannot close that gap —
it is already applied. Only an architectural change can. This script measures,
for each candidate architecture, exactly how much flash and activation RAM it
would need.

Why this needs no training
--------------------------
Flash footprint and activation arena are determined by the *architecture* — the
shapes of the weight tensors and the shapes of the intermediate tensors. They do
not depend on the values those weights take. So an untrained model quantizes to
exactly the same number of bytes as a trained one, and this whole sweep runs
without a single training step.

That splits Phase 6 cleanly in two:

    Stage 1 (this script)  what does each architecture COST?     no training
    Stage 2 (train.py)     what does each architecture BUY?      training

Only variants that survive Stage 1 are worth spending Stage 2 GPU time on.

The candidates
--------------
Parameter profiling of stock YOLOv8n (nc=1) showed weights are not spread
evenly across scales:

    P5/32 pathway (backbone 7,8,9 + head 19,21)   51.8%
    Detect head                                    25.0%
    P3+P4 feature trunk                            23.2%

Over half the network exists to detect large objects. P5/32 fires on objects
roughly >=320px in a 640px frame — a person filling half the frame height. In
fixed-mount overhead CCTV that essentially never happens, so on this domain that
half of the model is capacity the deployment never uses.

So the sweep crosses two levers:

    width   the blunt instrument: scale every channel count (params ~ width^2)
    scales  remove the P5/32 detection branch entirely (configs/yolov8-p34.yaml)

The interesting comparison is the matched-budget pair — uniform width 0.125 and
the P5-free model at full width land within ~6% of the same parameter count, but
spend those parameters very differently. Stage 2 decides which spend is better.

Usage:
    python ablate.py
    python ablate.py --variants baseline p34
    python ablate.py --skip-arena          # flash only, fast
"""
import argparse
import contextlib
import io
import json
import logging
import os
import shutil
import tempfile
import warnings
from pathlib import Path

os.environ.setdefault("YOLO_VERBOSE", "False")

import yaml

from mcu_common import MCUS, profile_arena, weight_footprint

parser = argparse.ArgumentParser()
parser.add_argument("--calib", default="data/yolo_dataset/calibration.yaml",
                    help="Calibration yaml for INT8 export (reuses the leak-free Phase 4 set)")
parser.add_argument("--imgsz", type=int, default=640)
parser.add_argument("--cache-dir", default="models/ablation_cache")
parser.add_argument("--runtime-overhead-kb", type=float, default=32.0,
                    help="RAM reserved for the inference runtime itself (default: 32 KB)")
parser.add_argument("--variants", nargs="+", default=None,
                    help="Subset of variant keys to run (default: all)")
parser.add_argument("--resolutions", type=int, nargs="+",
                    default=[640, 512, 416, 320, 256, 192, 160, 128],
                    help="Input resolutions for the deployment-envelope sweep")
parser.add_argument("--skip-envelope", action="store_true",
                    help="Skip the architecture x resolution envelope sweep")
parser.add_argument("--skip-arena", action="store_true",
                    help="Skip activation-RAM analysis; report flash only")
parser.add_argument("--output", default="results/phase6_ablation.json")
args = parser.parse_args()

SCALED = "configs/yolov8-scaled.yaml"
P34    = "configs/yolov8-p34.yaml"

# Labels are deliberately self-contained rather than tree-indented: the same
# row appears in several tables below, and "+ width 0.125" is ambiguous once
# separated from its parent.
# key            label                    cfg      scale  note
VARIANTS = [
    ("baseline",  "P3/P4/P5  w0.25",       SCALED,  "n", "Phase 3-5 baseline"),
    ("w1875",     "P3/P4/P5  w0.1875",     SCALED,  "t", "uniform 0.75x width"),
    ("w125",      "P3/P4/P5  w0.125",      SCALED,  "p", "uniform 0.50x width"),
    ("p34",       "P3/P4     w0.25",       P34,     "n", "P5 branch removed, full width"),
    ("p34w1875",  "P3/P4     w0.1875",     P34,     "t", "P5 removed + 0.75x width"),
    ("p34w125",   "P3/P4     w0.125",      P34,     "p", "P5 removed + 0.50x width"),
]

if args.variants:
    keep = set(args.variants)
    unknown = keep - {v[0] for v in VARIANTS}
    if unknown:
        print(f"Unknown variant(s): {', '.join(sorted(unknown))}")
        print(f"Available: {', '.join(v[0] for v in VARIANTS)}")
        raise SystemExit(1)
    VARIANTS = [v for v in VARIANTS if v[0] in keep]

if not Path(args.calib).exists():
    print(f"Calibration yaml not found: {args.calib}\nRun quantize.py first.")
    raise SystemExit(1)

warnings.filterwarnings("ignore")
# This script runs ~50 exports; ultralytics logs a Netron banner for each one,
# which buries the tables. Its output carries no information here.
logging.getLogger("ultralytics").setLevel(logging.ERROR)

CACHE = Path(args.cache_dir)
CACHE.mkdir(parents=True, exist_ok=True)


def resolve_cfg(cfg_path, scale, workdir):
    """Write a temp yaml with the requested scale baked in as the sole entry.

    Ultralytics resolves a model's scale from the *filename* via a regex that
    only recognises [nslmx], so the custom narrow rungs ('t', 'p') cannot be
    selected that way. Collapsing `scales` to a single entry keyed 'n' and
    naming the file yolov8n-*.yaml makes the selection unambiguous regardless
    of how the loader guesses.
    """
    base = yaml.safe_load(open(cfg_path))
    d = dict(base)
    d["scales"] = {"n": base["scales"][scale]}
    stem = f"yolov8n-{Path(cfg_path).stem}-{scale}"
    out = Path(workdir) / f"{stem}.yaml"
    yaml.safe_dump(d, open(out, "w"), sort_keys=False)
    return out, base["scales"][scale]


def build(cfg_path, scale, workdir):
    from ultralytics import YOLO
    resolved, dims = resolve_cfg(cfg_path, scale, workdir)
    with contextlib.redirect_stdout(io.StringIO()):
        model = YOLO(str(resolved))
    return model, dims


def export_int8(key, cfg_path, scale, workdir):
    """Export a variant to INT8 ONNX (used for FLASH only), cached by key."""
    dest = CACHE / f"{key}_int8.onnx"
    if dest.exists():
        return dest
    model, _ = build(cfg_path, scale, workdir)
    with contextlib.redirect_stdout(io.StringIO()):
        src = Path(model.export(format="onnx", imgsz=args.imgsz, quantize="int8",
                                data=args.calib, verbose=False))
    shutil.move(str(src), dest)
    # ultralytics leaves the intermediate FP32 graph beside the INT8 one
    stray = src.with_name(src.name.replace("_int8", ""))
    if stray.exists():
        stray.unlink()
    return dest


def export_fp32(key, cfg_path, scale, workdir, imgsz=None):
    """Export a variant to FP32 ONNX (used for ARENA only), cached by key+imgsz.

    Arena must be measured here rather than on the INT8 graph — see
    mcu_common.profile_arena. Activation shapes are identical either way, and
    this graph keeps the fusible Conv/Sigmoid/Mul topology intact.
    """
    imgsz = imgsz or args.imgsz
    dest = CACHE / f"{key}_fp32_{imgsz}.onnx"
    if dest.exists():
        return dest
    model, _ = build(cfg_path, scale, workdir)
    with contextlib.redirect_stdout(io.StringIO()):
        src = Path(model.export(format="onnx", imgsz=imgsz, verbose=False))
    shutil.move(str(src), dest)
    return dest


# Anchor/stride constants are three float32 tensors over all anchor points:
# two of shape (1,2,N) and one of shape (1,N), i.e. 20 bytes per anchor point.
# N is the sum of the feature-map cell counts over the detection strides, so it
# scales with resolution^2 while the convolution weights do not. Verified
# against measurement below.
ANCHOR_BYTES_PER_POINT = 20


def anchor_bytes(imgsz, strides):
    return ANCHOR_BYTES_PER_POINT * sum((imgsz // s) ** 2 for s in strides)


def param_count(cfg_path, scale, workdir):
    model, dims = build(cfg_path, scale, workdir)
    n = sum(p.numel() for p in model.model.parameters())
    strides = [int(s) for s in model.model.stride.tolist()]
    return n, dims, strides


# ---------------------------------------------------------------- run
print("Phase 6 — architecture ablation against the MCU flash budget")
print(f"  target: 2,048 KB flash (STM32H743, largest plain STM32)")
print(f"  imgsz={args.imgsz}  calib={args.calib}\n")

rows = []
with tempfile.TemporaryDirectory() as workdir:
    for key, label, cfg_path, scale, note in VARIANTS:
        print(f"--- {label.strip()} ({key}) ---")
        n_params, dims, strides = param_count(cfg_path, scale, workdir)
        print(f"  params : {n_params:,}   (depth {dims[0]}, width {dims[1]}, "
              f"strides {strides})")

        int8_path = export_int8(key, cfg_path, scale, workdir)
        flash = weight_footprint(int8_path)
        print(f"  flash  : {flash['total_kb']:.1f} KB "
              f"(weights {flash['weights_only_kb']:.1f} + anchors {flash['anchor_const_kb']:.1f})")

        # Validate the analytic anchor model against measurement before the
        # envelope sweep relies on it to extrapolate to other resolutions.
        predicted = anchor_bytes(args.imgsz, strides) / 1024
        if abs(predicted - flash["anchor_const_kb"]) > 0.5:
            print(f"  !! anchor model mismatch: predicted {predicted:.1f} KB, "
                  f"measured {flash['anchor_const_kb']:.1f} KB")
            raise SystemExit(1)

        row = {
            "key": key, "label": label.strip(), "cfg": cfg_path, "scale": scale,
            "note": note, "depth_mult": dims[0], "width_mult": dims[1],
            "params": n_params, "strides": strides, "flash": flash,
        }

        if not args.skip_arena:
            # FP32 graph, deliberately — the INT8 QDQ graph would inflate this
            # ~2x by defeating SiLU fusion and counting Q/DQ intermediates.
            fp32_path = export_fp32(key, cfg_path, scale, workdir)
            prof = profile_arena(fp32_path, act_bytes_per_elem=1, fuse=True)
            prof["arena_kb_unfused"] = profile_arena(
                fp32_path, act_bytes_per_elem=1, fuse=False)["arena_kb"]
            row["arena"] = prof
            print(f"  arena  : {prof['arena_kb']:.1f} KB  "
                  f"(peak node {prof['peak_at_node']} {prof['peak_node_op']})")
        print()
        rows.append(row)

# ---------------------------------------------------------------- table
base_row = next((r for r in rows if r["key"] == "baseline"), None)

print("\n=== Phase 6 — cost of each architecture (no training) ===\n")
head = f"{'Variant':<22} {'Params':>10} {'Flash KB':>9} {'vs base':>8}"
if not args.skip_arena:
    head += f" {'Arena KB':>9}"
head += f"  {'H743':>5}"
print(head)
print("-" * len(head))

H743_FLASH, H743_RAM = 2048, 1024
for r in rows:
    ratio = (r["flash"]["total_kb"] / base_row["flash"]["total_kb"]) if base_row else float("nan")
    fits = (r["flash"]["total_kb"] <= H743_FLASH and
            (args.skip_arena or r["arena"]["arena_kb"] + args.runtime_overhead_kb <= H743_RAM))
    line = f"{r['label']:<22} {r['params']:>10,} {r['flash']['total_kb']:>9.1f} {ratio:>7.2f}x"
    if not args.skip_arena:
        line += f" {r['arena']['arena_kb']:>9.1f}"
    line += f"  {'OK' if fits else 'over':>5}"
    print(line)

print(f"\n(H743 = 2,048 KB flash / 1,024 KB RAM; RAM need = arena + "
      f"{args.runtime_overhead_kb:.0f} KB runtime)")

# ---------------------------------------------------------------- full feasibility
print(f"\n\n=== Feasibility across targets, at {args.imgsz}x{args.imgsz} ===\n")
print("(resolution is held fixed here so the ARCHITECTURE is the only variable;\n"
      " the envelope section below then varies resolution too)\n")
name_w = max(len(m[0]) for m in MCUS)
header = f"{'MCU':<{name_w}} {'Flash':>7} {'RAM':>7}  | " + "  ".join(f"{r['key']:>9}" for r in rows)
print(header)
print("-" * len(header))

feasibility = {}
for name, fkb, rkb, npu, note in MCUS:
    cells, per_variant = [], {}
    for r in rows:
        flash_ok = r["flash"]["total_kb"] <= fkb
        ram_ok = args.skip_arena or (r["arena"]["arena_kb"] + args.runtime_overhead_kb) <= rkb
        fits = flash_ok and ram_ok
        cells.append("OK" if fits else ("ram" if flash_ok else "-"))
        per_variant[r["key"]] = {"fits": fits, "flash_ok": flash_ok, "ram_ok": ram_ok}
    feasibility[name] = {"flash_kb": fkb, "ram_kb": rkb, "has_npu": npu,
                         "note": note, "by_variant": per_variant}
    print(f"{name:<{name_w}} {fkb:>7} {rkb:>7}  | " + "  ".join(f"{c:>9}" for c in cells))

print("\n  OK = fits    ram = flash fits but RAM does not    -  = flash too small")

# ---------------------------------------------------------------- matched budget
matched = [r for r in rows if r["key"] in ("w125", "p34")]
if len(matched) == 2:
    a, b = matched
    print("\n\n--- Matched-parameter comparison (the Stage 2 experiment) ---")
    print(f"  {a['label']:<22} {a['params']:>10,} params   {a['flash']['total_kb']:>7.1f} KB flash")
    print(f"  {b['label']:<22} {b['params']:>10,} params   {b['flash']['total_kb']:>7.1f} KB flash")
    print(f"\n  Within {abs(a['params'] - b['params']) / max(a['params'], b['params']) * 100:.1f}% "
          f"on parameter count, but spent differently:")
    print(f"    {a['label'].strip()} keeps all three scales at half channel width.")
    print(f"    {b['label'].strip()} keeps full channel width at the two scales")
    print(f"      surveillance targets actually occupy, and drops the third.")
    print("\n  Stage 2 (training) decides which spend buys more mAP. Flash says")
    print("  they cost the same, so this is a fair architectural question.")

# ---------------------------------------------------------------- envelope
# The two constraints turn out to be driven by different things:
#
#   flash  <- architecture   (weights; resolution only moves the anchor consts)
#   arena  <- resolution     (peak is the input buffer + first conv output,
#                             and the input buffer is architecture-independent)
#
# Phase 5 concluded "flash is the binding constraint, not RAM" — true for the
# stock architecture, where flash was so far over budget that RAM never got to
# matter. Once architecture fixes flash, RAM becomes binding, and no
# architectural change touches it. So neither lever alone is sufficient and the
# real deployment question is two-dimensional.
envelope = {}
if not args.skip_envelope:
    print("\n\n=== Deployment envelope: architecture x resolution ===\n")
    print("Arena KB (fused INT8, lower bound):\n")
    res = sorted(args.resolutions, reverse=True)
    hdr = f"{'Variant':<22} " + " ".join(f"{r:>6}" for r in res)
    print(hdr)
    print("-" * len(hdr))

    with tempfile.TemporaryDirectory() as workdir:
        for r in rows:
            cells = []
            per_res = {}
            for imgsz in res:
                fp32 = export_fp32(r["key"], r["cfg"], r["scale"], workdir, imgsz=imgsz)
                prof = profile_arena(fp32, act_bytes_per_elem=1, fuse=True)
                fl_kb = r["flash"]["weights_only_kb"] + anchor_bytes(imgsz, r["strides"]) / 1024
                per_res[imgsz] = {
                    "arena_kb": prof["arena_kb"],
                    "flash_kb": round(fl_kb, 1),
                    "ram_need_kb": round(prof["arena_kb"] + args.runtime_overhead_kb, 1),
                }
                cells.append(f"{prof['arena_kb']:>6.0f}")
            envelope[r["key"]] = per_res
            print(f"{r['label']:<22} " + " ".join(cells))

    print("\n(arena is near-identical across architectures — the peak is the "
          "input\n buffer plus the first conv output, both set by resolution)")

    # ---- what actually deploys
    print("\n\n=== What actually deploys ===\n")
    for name, fkb, rkb, npu, note in MCUS:
        if npu:
            continue                       # the point of the project is to avoid these
        hits = []
        for r in rows:
            ok = [imgsz for imgsz in res
                  if envelope[r["key"]][imgsz]["flash_kb"] <= fkb
                  and envelope[r["key"]][imgsz]["ram_need_kb"] <= rkb]
            if ok:
                hits.append((r["label"], max(ok), envelope[r["key"]][max(ok)]))
        print(f"{name}  ({fkb} KB flash / {rkb} KB RAM)")
        if not hits:
            print("    nothing fits at any resolution tested")
        for label, imgsz, cell in hits:
            print(f"    {label:<22} up to {imgsz}x{imgsz}   "
                  f"flash {cell['flash_kb']:>6.1f} KB   RAM {cell['ram_need_kb']:>6.1f} KB")
        print()

out = Path(args.output)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({
    "method": ("Flash and activation arena are architecture-determined, not "
               "weight-determined, so every variant here is measured UNTRAINED "
               "and the numbers are exact. Flash read from INT8 ONNX "
               "initializers; arena via liveness analysis assuming a perfect "
               "memory planner and fused INT8 activations (lower bound). "
               "Accuracy is NOT measured here — see Stage 2."),
    "imgsz": args.imgsz,
    "runtime_overhead_kb": args.runtime_overhead_kb,
    "target": {"mcu": "STM32H743", "flash_kb": H743_FLASH, "ram_kb": H743_RAM},
    "variants": rows,
    "feasibility": feasibility,
    "envelope": envelope,
    "envelope_note": ("flash is set by architecture, arena by input resolution "
                      "(the peak is the input buffer plus the first conv output). "
                      "Neither lever alone is sufficient. Flash at a given "
                      "resolution = measured INT8 weights + analytic anchor "
                      "constants, the latter validated against measurement at "
                      f"{args.imgsz}px."),
}, indent=2))
print(f"\nSaved → {out}")
