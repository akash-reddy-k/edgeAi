"""
Phase 5 — MCU deployment feasibility.

Answers the question the whole project is built around: does a commercial-grade
person detector actually fit on a plain microcontroller with no AI accelerator?

Method
------
STM32Cube.AI and Edge Impulse EON both report two numbers that decide whether a
network can be deployed: **flash** (weights, stored once) and **activation RAM**
(the working buffer, reused across layers). This script computes both directly
from the ONNX graph, so the result is reproducible by anyone with the repo and
does not depend on a vendor account or GUI.

Both measurements live in mcu_common.py, shared with the Phase 6 ablation —
see that module for the liveness analysis and the SiLU fusion pass, and for why
the arena figure is deliberately a lower bound.

Usage:
    python mcu_profile.py
    python mcu_profile.py --resolutions 640 320 160 96
"""
import argparse
import json
from pathlib import Path

from ultralytics import YOLO

from mcu_common import MCUS, profile_arena, weight_footprint

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="models/yolov8n_finetuned.pt")
parser.add_argument("--int8-model", default="models/yolov8n_finetuned_int8.onnx")
parser.add_argument("--resolutions", type=int, nargs="+",
                    default=[640, 512, 416, 320, 256, 192, 160, 128, 96, 64])
parser.add_argument("--cache-dir", default="models/profile_cache")
parser.add_argument("--runtime-overhead-kb", type=float, default=32.0,
                    help="RAM reserved for the inference runtime itself (default: 32 KB)")
parser.add_argument("--output", default="results/phase5_mcu_profile.json")
args = parser.parse_args()


def export_at(imgsz, cache_dir):
    """Export FP32 ONNX at a given input resolution (topology only — we count
    activation elements, so FP32 vs INT8 export does not change the shapes)."""
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    dest = cache / f"yolov8n_{imgsz}.onnx"
    if dest.exists():
        return dest
    src = Path(YOLO(args.model).export(format="onnx", imgsz=imgsz, verbose=False))
    src.rename(dest)
    return dest


# ---------------------------------------------------------------- run
if not Path(args.int8_model).exists():
    print(f"Missing {args.int8_model}\nRun quantize.py first.")
    raise SystemExit(1)

print("Phase 5 — MCU deployment feasibility\n")

flash = weight_footprint(args.int8_model)
print(f"Flash (INT8 weights): {flash['total_kb']:.1f} KB  ({flash['total_kb']/1024:.2f} MB)")
for dt, kb in flash["by_dtype_kb"].items():
    print(f"    {dt:<10} {kb:>9.1f} KB")
print()

print("Profiling activation arena across input resolutions ...")
print(f"  {'input':>9}  {'fused KB':>10}  {'unfused KB':>11}   peak op\n")
rows = []
for imgsz in sorted(args.resolutions, reverse=True):
    path = export_at(imgsz, args.cache_dir)
    prof = profile_arena(path, act_bytes_per_elem=1, fuse=True)
    naive = profile_arena(path, act_bytes_per_elem=1, fuse=False)
    prof["arena_kb_unfused"] = naive["arena_kb"]
    rows.append({"imgsz": imgsz, **prof})
    print(f"  {imgsz:>4}x{imgsz:<4} {prof['arena_kb']:>10.1f}  {naive['arena_kb']:>11.1f}   "
          f"node {prof['peak_at_node']} {prof['peak_node_op']}")

# ---------------------------------------------------------------- feasibility
# Flash is held at its 640x640 value across every column. It is not perfectly
# constant: the weights are (2,952.8 KB), but the Detect head's anchor/stride
# constants are not, falling from 164.1 KB at 640 to 2.2 KB at 96. Holding the
# maximum fixed overstates flash by at most that 162 KB at the smallest input,
# which is the conservative direction for a negative feasibility claim and
# changes no verdict in the table below (the nearest miss is still 44% over).
# Phase 6's ablate.py models the resolution-dependent term explicitly, because
# on sub-megabyte models it stops being negligible.
print("\n\n=== Deployment feasibility ===")
print(f"(flash need = {flash['total_kb']:.0f} KB, held at its 640x640 maximum; "
      f"RAM need = arena + {args.runtime_overhead_kb:.0f} KB runtime)\n")

name_w = max(len(m[0]) for m in MCUS)
header = f"{'MCU':<{name_w}} {'Flash':>7} {'RAM':>7}  | " + "  ".join(f"{r['imgsz']:>4}" for r in rows)
print(header)
print("-" * len(header))

feasibility = {}
for name, fkb, rkb, npu, note in MCUS:
    flash_ok = flash["total_kb"] <= fkb
    cells, per_res = [], {}
    for r in rows:
        ram_ok = (r["arena_kb"] + args.runtime_overhead_kb) <= rkb
        fits = flash_ok and ram_ok
        cells.append(" OK " if fits else ("ram " if flash_ok else "  - "))
        per_res[r["imgsz"]] = {"fits": fits, "flash_ok": flash_ok, "ram_ok": ram_ok}
    feasibility[name] = {"flash_kb": fkb, "ram_kb": rkb, "has_npu": npu,
                         "note": note, "by_resolution": per_res}
    print(f"{name:<{name_w}} {fkb:>7} {rkb:>7}  | " + "  ".join(f"{c:>4}" for c in cells))

print("\n  OK = fits    ram = flash fits but RAM does not    -   = flash too small")

# Smallest resolution that fits nothing / the frontier
print("\n--- Frontier ---")
for name, fkb, rkb, npu, note in MCUS:
    ok = [r["imgsz"] for r in rows
          if flash["total_kb"] <= fkb and (r["arena_kb"] + args.runtime_overhead_kb) <= rkb]
    verdict = f"max {max(ok)}x{max(ok)}" if ok else "NO resolution fits"
    print(f"  {name:<{name_w}}  {verdict}")

out = Path(args.output)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({
    "method": ("Flash from INT8 ONNX initializers. Activation arena via liveness "
               "analysis assuming a perfect memory planner and fused INT8 "
               "activations (1 byte/element) — a LOWER BOUND on real RAM."),
    "runtime_overhead_kb": args.runtime_overhead_kb,
    "flash": flash,
    "arena_by_resolution": rows,
    "feasibility": feasibility,
}, indent=2))
print(f"\nSaved → {out}")
