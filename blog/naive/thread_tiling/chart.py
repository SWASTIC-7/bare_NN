import matplotlib.pyplot as plt


def main() -> None:
    n_vals = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    gflops_vals = [
        0.000735,
        0.013740,
        0.122959,
        0.653061,
        7.850503,
        57.740968,
        176.260887,
        384.939792,
        450.228856,
        466.681930,
        502.375859,
    ]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(11, 6), dpi=140)

    ax.plot(
        n_vals,
        gflops_vals,
        color="#0B7285",
        linewidth=3,
        marker="o",
        markersize=7,
        markerfacecolor="#FFD43B",
        markeredgecolor="#0B7285",
        markeredgewidth=1.4,
        label="Thread-Tiled PTX",
    )

    ax.fill_between(n_vals, gflops_vals, color="#99E9F2", alpha=0.25)

    for x, y in zip(n_vals, gflops_vals):
        if x >= 256:
            ax.annotate(
                f"{y:.1f}",
                xy=(x, y),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                color="#134E4A",
            )

    ax.set_title("Thread-Tiled PTX Scaling", fontsize=18, weight="bold", pad=14)
    ax.set_xlabel("Matrix Size N (NxN)", fontsize=12)
    ax.set_ylabel("Performance (GFLOPS)", fontsize=12)
    ax.set_xscale("log", base=2)
    ax.set_xticks(n_vals)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    ax.set_ylim(bottom=0)
    ax.legend(loc="lower right", frameon=True)
    ax.grid(True, which="major", linestyle="--", alpha=0.35)

    fig.tight_layout()
    fig.savefig("thread_tiling_perf.png", dpi=220, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
