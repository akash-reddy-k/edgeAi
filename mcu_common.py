"""
Shared MCU footprint analysis — used by mcu_profile.py (Phase 5) and
ablate.py (Phase 6).

Extracted so the two phases cannot drift apart. The SiLU fusion pass and the
liveness analysis below are the correctness-critical parts of the feasibility
claim; having two copies would eventually mean two different answers.

Both numbers an MCU deployment lives or dies by are computed here:

    flash  — weights, stored once, read from the INT8 model's initializers
    arena  — activation RAM, the working buffer reused across layers

Arena is a liveness analysis over the execution schedule:

    for each tensor:  birth = index of the node producing it
                      death = index of the last node consuming it
    working set at node i = sum of bytes of all tensors live across i
    arena = max working set over all i

This assumes a *perfect* memory planner that reuses every buffer the instant it
dies. Real runtimes do worse. The result is therefore a LOWER BOUND on real
activation RAM — the conservative direction for a feasibility claim: if the
lower bound does not fit, the network definitively does not fit.

Activations are counted at 1 byte/element, modelling a fused INT8 deployment.
The exported QDQ graph carries float32 tensors between its Quantize/Dequantize
pairs, but an MCU runtime fuses those away; counting them would overstate RAM.
"""
from pathlib import Path

import onnx
from onnx import numpy_helper, shape_inference

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


class QDQGraphError(ValueError):
    """Raised when arena analysis is pointed at a quantized (QDQ) graph."""


def profile_arena(onnx_path, act_bytes_per_elem=1, fuse=True, allow_qdq=False):
    """Peak activation working set, via liveness analysis over the schedule.

    Must be given the **FP32** export, not the INT8 one. The INT8 graph is in
    QDQ form, which interposes QuantizeLinear/DequantizeLinear nodes between
    Conv -> Sigmoid -> Mul. That defeats fuse_silu() (the pattern no longer
    matches) and additionally counts the Q/DQ intermediates as live tensors.
    On YOLOv8n that inflates the arena 2.2x — 2,800 KB becomes 6,135 KB — and
    silently flips feasibility verdicts.

    Activation *shapes* are identical in both exports, so the FP32 graph is the
    correct thing to analyse; the INT8 graph is only used for flash.
    """
    model = shape_inference.infer_shapes(onnx.load(onnx_path))
    g = model.graph

    if not allow_qdq:
        qdq = sum(1 for n in g.node
                  if n.op_type in ("QuantizeLinear", "DequantizeLinear"))
        if qdq:
            raise QDQGraphError(
                f"{onnx_path} is a QDQ (quantized) graph — {qdq} Quantize/"
                f"DequantizeLinear nodes found. Arena analysis must run on the "
                f"FP32 export or the result is inflated ~2x. Pass the FP32 "
                f"model, or allow_qdq=True if you really mean it."
            )

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


# Anchor/stride constants are baked into the Detect head as plain float32
# initializers, so they land in flash alongside the weights. They are large
# (three tensors over all anchor points: 164 KB at 640x640 for a P3/P4/P5 head)
# and, unlike the convolution weights, they scale with INPUT RESOLUTION rather
# than with width. Phase 6 separates them out because on a sub-megabyte model
# they stop being a rounding error and become ~15% of the flash bill.
ANCHOR_MIN_ELEMS = 100


def weight_footprint(int8_path):
    m = onnx.load(int8_path)
    by_dtype, total, anchor_bytes = {}, 0, 0
    for init in m.graph.initializer:
        a = numpy_helper.to_array(init)
        by_dtype[str(a.dtype)] = by_dtype.get(str(a.dtype), 0) + a.nbytes
        total += a.nbytes
        if a.dtype.name == "float32" and a.size >= ANCHOR_MIN_ELEMS:
            anchor_bytes += a.nbytes
    return {
        "total_bytes": total,
        "total_kb": round(total / 1024, 1),
        "by_dtype_kb": {k: round(v / 1024, 1)
                        for k, v in sorted(by_dtype.items(), key=lambda x: -x[1])},
        "anchor_const_kb": round(anchor_bytes / 1024, 1),
        "weights_only_kb": round((total - anchor_bytes) / 1024, 1),
        "file_kb": round(Path(int8_path).stat().st_size / 1024, 1),
    }
