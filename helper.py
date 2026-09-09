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


#!/usr/bin/env python3
"""Count the operations in our grouped-query FlashAttention pseudocode.

Run: python fa2_compute_count.py
     python fa2_compute_count.py --br 128 --bc 256 --sa1-rows 32 --sa2-rows 96

Defaults use the bandwidth-model choice Br=Bc=256, with 128 query rows on
each array. K16 and V16 are loaded ONCE per (layer, batch, KV head), then
reused by all G=4 Q heads and all their query tiles.

This counts attention operations, not projection/FFN GEMMs, elapsed cycles,
or host transfers. Q/K/V have already been projected and reside in DRAM.
One physical array command computes one output microtile, one reduction
chunk, and one digit pair. Our safe INT8 method uses 2 digits per operand
(4 pairs) and reduction chunks <=128. Command overlap and input preloading
change timing, not the number of those commands.

SA1 owns the first sa1_rows rows of each query block; SA2 owns the rest.
The same fixed row ranges are used for QK and PV. A short final query block
uses only its valid ranges; partial physical output tiles are padded.
DMA byte totals include Q/K/V/O tensor payloads only, in 16-bit storage.
Quantization scales and other metadata transfers are outside these totals.
"""

from collections import Counter
import argparse


# User's model hyperparameters.
B = 16       # Batch size
L = 32       # Layers
T = 2048     # Q length
S = 2048     # K/V length
D = 4096     # Model dimension
F = 14336    # FFN intermediate dimension; unused in this attention counter
N = 32       # Number of Q heads
K = 8        # Number of KV heads (not a matrix reduction length)
H = 128      # Head dimension
G = N // K   # Integer: 4 Q heads share one KV head
d_kv = K * H # 1024

Br = 256
Bc = 256
DIGIT_PAIRS = 4
REDUCTION_CHUNK = 128


def divup(a, b):
    return (a + b - 1) // b


def physical_commands(rows, array_rows, columns, reduction):
    """One command per padded output tile, reduction chunk, and digit pair."""
    return (divup(rows, array_rows)
            * divup(columns, 16)
            * divup(reduction, REDUCTION_CHUNK)
            * DIGIT_PAIRS)


def count_fa2(B=B, L=L, T=T, S=S, D=D, N=N, K=K, H=H,
              Br=Br, Bc=Bc, sa1_rows=None, sa2_rows=None, causal=True):
    """Count operations and DRAM payload bytes across ALL batches and layers.

    Notation (x means multiplication; ceil means round up):
        G: N / K                         Q heads sharing one KV head
        Tr: ceil(T / Br)                  query blocks per Q head
        Tc: ceil(S / Bc)                  key blocks per KV head
        P: number of visited (query block, key block) pairs per Q head

    Block-pair formula:
        Noncausal: P = Tr x Tc.
        Causal, T=S, Br=Bc, T divisible by Br: P = Tr x (Tr + 1) / 2.
        General causal case, with zero-based query block i:
            r_i = min(Br, T - i x Br)
            J_i = ceil(min(S, i x Br + r_i) / Bc)
            P = sum(J_i for i = 0, ..., Tr-1)
        For noncausal formulas below, use J_i = Tc.

    Operation counts (calls, NOT numbers of scalar elements):
        KV_head_groups:                 B x L x K
        DMA_load_K16:                   B x L x K
        DMA_load_V16:                   B x L x K
        VCPU_quantize_KV_and_pack_V:     B x L x K
        Q_heads_processed:              B x L x K x G = B x L x N

        DMA_load_Qi16:                  B x L x N x Tr
        VCPU_quantize_Q_and_init_Ulm:    B x L x N x Tr
        VCPU_normalize_and_encode_Oi16:  B x L x N x Tr
        DMA_store_Oi16:                 B x L x N x Tr

        QK_logical_products:            B x L x N x P
        PV_logical_products:            B x L x N x P
        VCPU_pack_QK_digits:            B x L x N x P
        VCPU_reconstruct_QK:            B x L x N x P
        VCPU_softmax_and_quantize_P:     B x L x N x P
        VCPU_pack_PV_digits:            B x L x N x P
        VCPU_reconstruct_PV:            B x L x N x P
        VCPU_update_Ul:                 B x L x N x P
        CPU_swap_m_buffer:              B x L x N x P

        DMA_transfers_total: 2 x B x L x K + 2 x B x L x N x Tr
        VCPU_launches_total: B x L x K + 2 x B x L x N x Tr
                             + 6 x B x L x N x P

    Physical array commands, when ALL query/key blocks have full size:
        Let A=16, R=sa1_rows for SA1; A=32, R=sa2_rows for SA2.
        QK: B x L x N x P x ceil(R/A) x ceil(Bc/16)
            x ceil(H/REDUCTION_CHUNK) x DIGIT_PAIRS
        PV: B x L x N x P x ceil(R/A) x ceil(H/16)
            x ceil(Bc/REDUCTION_CHUNK) x DIGIT_PAIRS

        Exact formulas including short final blocks:
            c_j = min(Bc, S - j x Bc)
            R_i = min(r_i, sa1_rows) for SA1
            R_i = r_i - min(r_i, sa1_rows) for SA2
            QK = B x L x N x sum_i sum_{j < J_i}
                 physical_commands(R_i, A, c_j, H)
            PV = B x L x N x sum_i sum_{j < J_i}
                 physical_commands(R_i, A, H, c_j)
        SA1_commands_total: SA1_QK_commands + SA1_PV_commands
        SA2_commands_total: SA2_QK_commands + SA2_PV_commands
        SA_commands_total: SA1_commands_total + SA2_commands_total
        SA_operand_preloads_total: SA_commands_total (both operands per call)
        SA_partial_output_writes_total: SA_commands_total

    DRAM payload formulas (BYTES; 16-bit storage = 2 bytes per element):
        K16_read:  B x L x K x S x H x 2
        V16_read:  B x L x K x S x H x 2
        Q16_read:  B x L x N x T x H x 2 = B x L x T x D x 2
        O16_write: B x L x N x T x H x 2 = B x L x T x D x 2
        Combined K/V read: B x L x S x d_kv x 4, where d_kv = K x H.

    K/V counts have NO G or Tr factor: the complete KV head stays in SRAM
    and is reused by its G grouped Q heads and all their query blocks.
    These formulas describe this counter's loop/packing choices, including
    its four digit pairs; they are not hardware-required launch counts.
    Returns: (operation_counts, payload_bytes), both Counter objects.
    """
    if any(not isinstance(x, int) or x <= 0
           for x in (B, L, T, S, D, N, K, H, Br, Bc)):
        raise ValueError("Dimensions must be positive integers")
    if N % K or D != N * H:
        raise ValueError("Require N divisible by K, and D = N * H")
    if causal and T != S:
        raise ValueError("Causal mode here models aligned self-attention with T=S")
    if sa1_rows is None and sa2_rows is None:
        sa1_rows = sa2_rows = Br // 2
    if (sa1_rows is None or sa2_rows is None or sa1_rows < 0 or sa2_rows < 0
            or sa1_rows + sa2_rows != Br
            or sa1_rows % 16 or sa2_rows % 32):
        raise ValueError("Choose SA1 rows divisible by 16 and SA2 rows divisible by 32; their sum must equal Br")

    G = N // K
    count = Counter()
    payload_bytes = Counter()

    for layer in range(L):
        for batch in range(B):
            for kv_head in range(K):
                # OUTSIDE the G loop: load K/V once for four grouped Q heads.
                count["KV_head_groups"] += 1
                count["DMA_load_K16"] += 1
                count["DMA_load_V16"] += 1
                payload_bytes["K16_read"] += S * H * 2
                payload_bytes["V16_read"] += S * H * 2
                count["VCPU_quantize_KV_and_pack_V"] += 1

                for g in range(G):
                    q_head = kv_head * G + g
                    assert 0 <= q_head < N
                    count["Q_heads_processed"] += 1

                    for q0 in range(0, T, Br):
                        r = min(Br, T - q0)
                        r1 = min(r, sa1_rows)
                        r2 = r - r1

                        count["DMA_load_Qi16"] += 1
                        payload_bytes["Q16_read"] += r * H * 2
                        count["VCPU_quantize_Q_and_init_Ulm"] += 1

                        # Skip whole key blocks strictly above the causal
                        # diagonal. A visited block is computed in full;
                        # individual future entries are masked afterward.
                        key_stop = min(S, q0 + r) if causal else S
                        for k0 in range(0, key_stop, Bc):
                            c = min(Bc, S - k0)

                            count["QK_logical_products"] += 1
                            count["VCPU_pack_QK_digits"] += 1
                            count["SA1_QK_commands"] += physical_commands(r1, 16, c, H)
                            count["SA2_QK_commands"] += physical_commands(r2, 32, c, H)
                            count["VCPU_reconstruct_QK"] += 1
                            count["VCPU_softmax_and_quantize_P"] += 1

                            count["PV_logical_products"] += 1
                            count["VCPU_pack_PV_digits"] += 1
                            count["SA1_PV_commands"] += physical_commands(r1, 16, H, c)
                            count["SA2_PV_commands"] += physical_commands(r2, 32, H, c)
                            count["VCPU_reconstruct_PV"] += 1
                            count["VCPU_update_Ul"] += 1
                            count["CPU_swap_m_buffer"] += 1

                        count["VCPU_normalize_and_encode_Oi16"] += 1
                        count["DMA_store_Oi16"] += 1
                        payload_bytes["O16_write"] += r * H * 2

    # Totals below are derived; do not add them to their component counts.
    count["VCPU_launches_total"] = sum(v for name, v in count.items() if name.startswith("VCPU_"))
    count["DMA_transfers_total"] = sum(v for name, v in count.items() if name.startswith("DMA_"))
    count["SA1_commands_total"] = count["SA1_QK_commands"] + count["SA1_PV_commands"]
    count["SA2_commands_total"] = count["SA2_QK_commands"] + count["SA2_PV_commands"]
    count["SA_commands_total"] = count["SA1_commands_total"] + count["SA2_commands_total"]
    # Each physical command has one operand-preload operation (two operands)
    # and one INT16 partial-output write. These are SRAM/array transfers,
    # not additional DRAM DMA commands.
    count["SA_operand_preloads_total"] = count["SA_commands_total"]
    count["SA_partial_output_writes_total"] = count["SA_commands_total"]
    return count, payload_bytes




