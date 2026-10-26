"""
Regenerates writeup/figures/*.pdf from the already-committed results JSON and
Stage 2 training logs. Run from the repo root:

    python writeup/make_figures.py

No new measurements are taken here — this script only re-plots numbers that
already exist in results/phase6_ablation.json and
runs/detect/results/training/yolov8_ablation_{w125,p34}/results.csv, so the
paper's figures and its claims cannot silently drift apart.
"""
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ---------------------------------------------------------------------------
# Figure 1 — flash footprint per architecture variant vs. MCU flash budgets
# ---------------------------------------------------------------------------
ablation = json.loads((ROOT / "results/phase6_ablation.json").read_text())
variants = ablation["variants"]

labels = [v["label"] for v in variants]
flash_kb = [v["flash"]["total_kb"] for v in variants]
colors = ["#4477AA" if "P5" not in v["label"] and v["strides"] == [8, 16] else "#CC6677"
          for v in variants]
# Simpler: color by whether P5 (stride 32) is present.
colors = ["#CC6677" if 32 in v["strides"] else "#4477AA" for v in variants]

fig, ax = plt.subplots(figsize=(6.4, 3.6))
bars = ax.bar(range(len(labels)), flash_kb, color=colors)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=20, ha="right")
ax.set_ylabel("INT8 flash footprint (KB)")
ax.set_title("Flash cost by architecture (640×640, measured)")

mcu_budgets = [
    ("STM32F746", 1024),
    ("STM32F767 / H743", 2048),
]
for name, budget in mcu_budgets:
    ax.axhline(budget, color="gray", linestyle="--", linewidth=1)
    ax.text(len(labels) - 0.4, budget + 40, f"{name} ({budget} KB)",
            fontsize=8, color="gray", ha="right")

from matplotlib.patches import Patch
legend_handles = [
    Patch(facecolor="#CC6677", label="P3/P4/P5 (3 scales)"),
    Patch(facecolor="#4477AA", label="P3/P4 (P5 removed)"),
]
ax.legend(handles=legend_handles, loc="upper right", frameon=False)

fig.tight_layout()
fig.savefig(FIG_DIR / "flash_by_variant.pdf")
plt.close(fig)
print("wrote", FIG_DIR / "flash_by_variant.pdf")

# ---------------------------------------------------------------------------
# Figure 2 — Stage 2 training curves: mAP50-95 per epoch, w125 vs p34
# ---------------------------------------------------------------------------
def read_curve(variant):
    path = ROOT / f"runs/detect/results/training/yolov8_ablation_{variant}/results.csv"
    with open(path) as f:
        rows = list(csv.DictReader(f))
    epochs = [int(r["epoch"]) for r in rows]
    map5095 = [float(r["metrics/mAP50-95(B)"]) for r in rows]
    map50 = [float(r["metrics/mAP50(B)"]) for r in rows]
    return epochs, map50, map5095

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.2), sharex=True)
for variant, color, label in [
    ("w125", "#CC6677", "P3/P4/P5 w0.125"),
    ("p34", "#4477AA", "P3/P4 w0.25"),
]:
    epochs, map50, map5095 = read_curve(variant)
    ax1.plot(epochs, map50, color=color, label=label, marker="o", markersize=2)
    ax2.plot(epochs, map5095, color=color, label=label, marker="o", markersize=2)

ax1.set_title("mAP50")
ax2.set_title("mAP50-95")
for ax in (ax1, ax2):
    ax.set_xlabel("epoch")
ax1.set_ylabel("validation score")
ax1.legend(frameon=False, fontsize=8, loc="lower right")

fig.tight_layout()
fig.savefig(FIG_DIR / "stage2_training_curves.pdf")
plt.close(fig)
print("wrote", FIG_DIR / "stage2_training_curves.pdf")
