import matplotlib.pyplot as plt
import matplotlib.patches as FancyBboxPatch
import numpy as np
from matplotlib.patches import FancyBboxPatch
import subprocess

def run(cmd): 
    out = subprocess.check_output(cmd,shell=True) 
    return float(out.decode().strip())

cuda_perf = run("./cuda")
triton_perf = run("python warp_triton.py")
ptx_perf = run("./warp_ptx")
cublas_perf = run("./cublas")

labels = ["CUDA warp tiled", "PTX warp tiled", "Triton warp tiled"]
values = [cuda_perf, ptx_perf, triton_perf]

colors = ["#C5A7E0", "#C5A7E0", "#C5A7E0", "#9882BD"]

fig, ax = plt.subplots(figsize=(8, 3.5))

bar_height = 0.45
y_positions = np.arange(len(labels))
max_val = max(values) 

def draw_rounded_bar(ax, x_start, y_center, width, height, color, alpha=1.0, radius=1):
    """Draw a rounded rectangle bar using FancyBboxPatch."""
    fancy = FancyBboxPatch(
        (x_start, y_center - height / 2),
        width,
        height,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=0,
        facecolor=color,
        alpha=alpha,
        zorder=2 if alpha == 1.0 else 1
    )
    ax.add_patch(fancy)

for i, (val, y) in enumerate(zip(values, y_positions)):
    # Ghost/track bar
    draw_rounded_bar(ax, 0, y, max_val * 1.0, bar_height, colors[i], alpha=0.15)
    # Main bar
    draw_rounded_bar(ax, 0, y, val, bar_height, colors[i], alpha=1.0)
    # Value label
    ax.text(
        val + max_val * 0.015,
        y,
        f"{val:,.0f}",
        va="center", ha="left",
        fontsize=10, color="#555555"
    )

ax.set_yticks(y_positions)
ax.set_yticklabels(labels, fontsize=11, color="#7C5CB4")
ax.set_xlim(0, max_val * 1.2)
ax.set_ylim(-0.6, len(labels) - 0.4)

# ax.xaxis.grid(True, color="#e0e0e0", linewidth=0.8, zorder=0)
ax.set_axisbelow(True)

for spine in ["top", "right", "left"]:
    ax.spines[spine].set_visible(False)
ax.spines["bottom"].set_color("#cccccc")

ax.tick_params(axis="x", colors="#888888", labelsize=9)
ax.tick_params(axis="y", left=False)
plt.xlabel("GFLOPS", fontsize=12)
plt.tight_layout()
plt.savefig("perf_chart.png", dpi=150, bbox_inches="tight")
plt.show()