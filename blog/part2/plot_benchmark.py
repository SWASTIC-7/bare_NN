"""Reusable bar chart for method-vs-GFLOP/s comparisons.

Horizontal rounded bars with a ghost track, matching the part-1 blog style.
Feed it {label: gflops} and it writes a PNG.  Deps: matplotlib, numpy.
"""


def plot_bars(results, out_path="benchmark.png", xlabel="GFLOPS",
              bar_color="#C5A7E0", label_color="#7C5CB4", radius=1):
    """
    results : dict {label: value}  (drawn top-to-bottom in insertion order)
    out_path: PNG path
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import FancyBboxPatch

    labels = list(results.keys())
    values = [results[k] for k in labels]

    fig, ax = plt.subplots(figsize=(8, 3.5))
    bar_height = 0.45
    y_positions = np.arange(len(labels))
    max_val = max(values) if values else 1.0

    def draw_rounded_bar(x_start, y_center, width, height, color, alpha=1.0):
        fancy = FancyBboxPatch(
            (x_start, y_center - height / 2), width, height,
            boxstyle=f"round,pad=0,rounding_size={radius}",
            linewidth=0, facecolor=color, alpha=alpha,
            zorder=2 if alpha == 1.0 else 1)
        ax.add_patch(fancy)

    for val, y in zip(values, y_positions):
        draw_rounded_bar(0, y, max_val * 1.0, bar_height, bar_color, alpha=0.15)  # track
        draw_rounded_bar(0, y, val, bar_height, bar_color, alpha=1.0)             # bar
        ax.text(val + max_val * 0.015, y, f"{val:,.0f}",
                va="center", ha="left", fontsize=10, color="#555555")

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=11, color=label_color)
    ax.set_xlim(0, max_val * 1.2)
    ax.set_ylim(-0.6, len(labels) - 0.4)
    ax.set_axisbelow(True)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#cccccc")
    ax.tick_params(axis="x", colors="#888888", labelsize=9)
    ax.tick_params(axis="y", left=False)
    ax.set_xlabel(xlabel, fontsize=12)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
