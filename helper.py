"""Plotting helpers for the report notebooks."""

import numpy as np
import matplotlib.pyplot as plt

# Categorical hues, assigned in fixed order (never cycled).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK_MUTED, GRID, AXIS = "#3d3d3a", "#6b6b64", "#e6e6e0", "#c9c9c1"


def array_ops_per_cycle(rows, K, bw_in, cols=16, out_bytes=2, bw_out=8):
    """Peak op/cycle of one systolic array at contraction depth K.

    The array is pipelined, so a tile op costs whichever port is slower:
        cost = max(input bytes / bw_in, output bytes / bw_out)
    Returns ops (2 per MAC), matching the FLOP convention used for the model.
    """
    cycles = max((rows + cols) * K / bw_in,          # operands in
                 rows * cols * out_bytes / bw_out)   # results out
    return 2 * rows * cols * K / cycles


def plot_roofline(points, peak_ops, peak_mem, clock=1.0, title=None,
                  xlim=(1, 1e4), figsize=(7.5, 4.8), savepath=None):
    """Draw a log-log roofline.

    points   : list of (label, arithmetic_intensity) in ops per DRAM byte
    peak_ops : peak compute, op/cycle
    peak_mem : peak DRAM bandwidth, byte/cycle
    clock    : GHz, so op/cycle reads directly as GOP/s
    """
    ridge = peak_ops / peak_mem

    ai = np.logspace(np.log10(xlim[0]) - 1, np.log10(xlim[1]), 500)
    roof = np.minimum(peak_ops * clock, ai * peak_mem * clock)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.plot(ai, roof, color=INK, lw=2, zorder=3)
    ax.fill_between(ai, 1, roof, color=INK, alpha=0.05, zorder=0)

    ax.axvline(ridge, color="#8c8c85", lw=1, ls="--", zorder=1)
    ax.annotate(f"ridge = {ridge:.0f} op/byte",
                xy=(ridge, peak_ops), xytext=(ridge * 1.25, peak_ops * 0.28),
                fontsize=9, color=INK)

    ax.text(xlim[0] * 1.25, peak_mem * xlim[0] * 0.85,
            f"memory bound\nslope = {peak_mem * clock:.0f} GB/s",
            fontsize=9, color=INK_MUTED, va="bottom")
    ax.text(xlim[1] * 0.92, peak_ops * 1.5,
            f"compute bound  ({peak_ops * clock / 1e3:.2f} TOPS)",
            fontsize=9, color=INK_MUTED, ha="right")

    for (label, x), c in zip(points, SERIES):
        y = min(peak_ops, x * peak_mem) * clock
        ax.plot([x, x], [1, y], color=c, lw=1, ls=":", alpha=0.6, zorder=2)
        ax.plot(x, y, "o", ms=9, color=c, mec="white", mew=2, zorder=4)
        dy = -34 if y >= peak_ops * 0.95 else 14      # keep clear of the roof
        ax.annotate(f"{label}\n{x:g} op/byte",
                    xy=(x, y), xytext=(0, dy), textcoords="offset points",
                    ha="center", fontsize=9.5, color=c, fontweight="bold")

    ax.set_xlim(*xlim)
    ax.set_ylim(peak_mem * xlim[0] * 0.5, peak_ops * 3)
    ax.set_xlabel("Arithmetic intensity  (op / DRAM byte)")
    ax.set_ylabel("Attainable performance  (GOP/s)")
    ax.set_title(title or f"Roofline — {peak_ops * clock / 1e3:.2f} TOPS / "
                          f"{peak_mem * clock:.0f} GB/s",
                 fontsize=12, loc="left", pad=14)
    ax.grid(True, which="major", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)

    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=160, facecolor="white")
    return fig, ax


from PIL import Image
from IPython.display import display

def plot(img, width):
    img = Image.open(img)
    height = round(img.height * width / img.width)
    display(img.resize((width, height)))