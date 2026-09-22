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

Flash is read from the INT8 model's actual initializers.

Activation RAM is a liveness analysis over the execution schedule:

    for each tensor:  birth = index of the node producing it
                      death = index of the last node consuming it
    working set at node i = sum of bytes of all tensors live across i
    arena = max working set over all i

This assumes a *perfect* memory planner that reuses every buffer the instant it
dies. Real runtimes do worse. The number below is therefore a **lower bound** on
activation RAM — which is the conservative direction for a feasibility claim: if
the lower bound does not fit, the network definitively does not fit.

Activations are counted at 1 byte/element, modelling a fused INT8 deployment.
The exported QDQ graph carries float32 tensors between its Quantize/Dequantize
pairs, but an MCU runtime fuses those away; counting them would overstate RAM.

Usage:
    python mcu_profile.py
    python mcu_profile.py --resolutions 640 320 160 96
"""
import argparse
import json
from pathlib import Path

import onnx
from onnx import numpy_helper, shape_inference
from ultralytics import YOLO

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

# Representative MCU targets. Flash/RAM in KB. "Plain" = no NPU.
MCUS = [
    # name,                 flash_kb, ram_kb,  has_npu, note
    ("STM32F746 (M7 216MHz)",   1024,    320,  False, "classic mid-range"),
    ("STM32F767 (M7 216MHz)",   2048,    512,  False, "high-end F7"),
    ("STM32H743 (M7 480MHz)",   2048,   1024,  False, "top plain STM32"),
    ("ESP32-S3 (LX7 240MHz)",   8192,    512,  False, "internal SRAM only"),
    ("ESP32-S3 + 8MB PSRAM",    8192,   8192,  False, "external PSRAM, slow"),
    ("STM32N657 (M55 + NPU)",   8192,   4300,  True,  "Neural-ART accelerator"),
]


def elem_count(vi):
    n = 1
    for d in vi.type.tensor_type.shape.dim:
        if not d.dim_value:
            return None
        n *= d.dim_value
    return n


def fuse_silu(nodes):
    """Collapse the ONNX SiLU idiom (y=Conv(x); s=Sigmoid(y); out=Mul(y,s)) into
    one node.

    This matters more than it looks. ONNX stores SiLU unfused, so at the Mul the
    naive analysis sees THREE full-size tensors live at once (y, s, out). Every
    real MCU runtime — CMSIS-NN, TFLite Micro, Cube.AI — emits a single fused
    activation kernel that computes out = y*sigmoid(y) in place over y, needing
    one buffer. Without this pass the arena is overstated ~3x at the peak, and
    since YOLOv8's peak sits on the very first SiLU, that error lands directly on
    the headline feasibility number.
    """
    produced_by = {}
    consumed_by = {}
    for i, (op, ins, outs) in enumerate(nodes):
        for o in outs:
            produced_by[o] = i
        for t in ins:
            consumed_by.setdefault(t, []).append(i)

    drop, rewire = set(), {}
    for i, (op, ins, outs) in enumerate(nodes):
        if op != "Sigmoid" or not ins or not outs:
            continue
        y, s = ins[0], outs[0]
        if y not in produced_by or len(consumed_by.get(s, [])) != 1:
            continue
        m = consumed_by[s][0]
        m_op, m_ins, m_outs = nodes[m]
        if m_op != "Mul" or set(m_ins) != {y, s}:
            continue
        # y must feed only the Sigmoid and the Mul, or it is needed elsewhere
        if sorted(consumed_by.get(y, [])) != sorted([i, m]):
            continue
        drop.update({i, m})
        rewire[produced_by[y]] = m_outs[0]     # producer writes the fused result

    fused = []
    for i, (op, ins, outs) in enumerate(nodes):
        if i in drop:
            continue
        if i in rewire:
            fused.append((op + "+SiLU", ins, [rewire[i]]))
        else:
            fused.append((op, ins, outs))
    return fused


def profile_arena(onnx_path, act_bytes_per_elem=1, fuse=True):
    """Peak activation working set, via liveness analysis over the schedule."""
    model = shape_inference.infer_shapes(onnx.load(onnx_path))
    g = model.graph

    weights = {init.name for init in g.initializer}
    sizes = {}
    for vi in list(g.value_info) + list(g.input) + list(g.output):
        if vi.name in weights:
            continue
        n = elem_count(vi)
        if n is not None:
            sizes[vi.name] = n * act_bytes_per_elem

    nodes = [(n.op_type, [t for t in n.input if t and t not in weights], list(n.output))
             for n in g.node]
    if fuse:
        nodes = fuse_silu(nodes)

    n_nodes = len(nodes)
    birth, death = {}, {}

    for name in (i.name for i in g.input):
        birth[name] = -1                      # input buffer exists before node 0
    for idx, (op, ins, outs) in enumerate(nodes):
        for out in outs:
            birth.setdefault(out, idx)
        for inp in ins:
            death[inp] = idx                  # nodes are topologically ordered
    for out in (o.name for o in g.output):
        death[out] = n_nodes                  # graph outputs live to the end

    peak, peak_at, peak_live = 0, -1, []
    for i in range(n_nodes):
        live = [t for t in sizes
                if birth.get(t, n_nodes + 1) <= i <= death.get(t, -1)]
        total = sum(sizes[t] for t in live)
        if total > peak:
            peak, peak_at, peak_live = total, i, live

    biggest = sorted(((sizes[t], t) for t in peak_live), reverse=True)[:5]
    return {
        "arena_bytes": peak,
        "arena_kb": round(peak / 1024, 1),
        "peak_at_node": peak_at,
        "peak_node_op": nodes[peak_at][0] if peak_at >= 0 else None,
        "live_tensors_at_peak": len(peak_live),
        "largest_live": [{"kb": round(b / 1024, 1), "name": t} for b, t in biggest],
        "n_nodes": n_nodes,
        "silu_fused": fuse,
    }


def weight_footprint(int8_path):
    m = onnx.load(int8_path)
    by_dtype, total = {}, 0
    for init in m.graph.initializer:
        a = numpy_helper.to_array(init)
        by_dtype[str(a.dtype)] = by_dtype.get(str(a.dtype), 0) + a.nbytes
        total += a.nbytes
    return {
        "total_bytes": total,
        "total_kb": round(total / 1024, 1),
        "by_dtype_kb": {k: round(v / 1024, 1) for k, v in sorted(by_dtype.items(), key=lambda x: -x[1])},
        "file_kb": round(Path(int8_path).stat().st_size / 1024, 1),
    }


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
print("\n\n=== Deployment feasibility ===")
print(f"(flash need = {flash['total_kb']:.0f} KB constant; "
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
