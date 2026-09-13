from math import ceil

from helper import *
from tile_model import *

MB = 2 ** 20
GB = 2 ** 30

## Latency model
BW_DMA = 64                                     # DRAM -> DMA -> SRAM
BW_VCPU = 128                                   # SRAM <-> vector CPU
PORTS = {"sa16": [(16, 64, 8)],                 # (rows, in B/cyc, out B/cyc)
         "sa32": [(32, 96, 8)],
         "both": [(16, 64, 8), (32, 96, 8)]}


def divup(a, b):
    return (a + b - 1) // b


def physical_commands(rows, array_rows, columns, reduction):
    """One command per padded output tile, reduction chunk, and digit pair."""
    return (divup(rows, array_rows)
            * divup(columns, 16)
            * divup(reduction, REDUCTION_CHUNK)
            * DIGIT_PAIRS)


def kernel_time(x_bytes, unit="both", K1=256, K2=None):
    """
    Cycles from control-CPU launch until the result is back in DRAM.

    K1 : contraction depth of the first array (the only array, unless unit="both")
    K2 : contraction depth of the second array ("both" only; defaults to K1)
    """
    LAT = 200 + 200 + 200 + 150  # Control-> DMA, DRAM read, control->DMA, DRAM write
    if unit == "vcpu":
        y = x_bytes
        compute = (x_bytes + y) / BW_VCPU
        launch = 300
    else:
        if K2 is None:
            K2 = K1
        ports = PORTS[unit]
        Ks = [K1, K2][:len(ports)]              # a single array uses K1 only

        # geometry and cost of ONE tile op on each array
        tiles = []
        for (rows, in_bw, out_bw), K in zip(ports, Ks):
            tin, tout = (rows + 16) * K, rows * 16 * 2      # bytes per tile op, in and out
            # pipelined array -> a tile costs max(in, out), never in + out
            cost = max(tin / in_bw, tout / out_bw)
            tiles.append((tin, tout, cost, tin / cost))     # tin/cost = EFFECTIVE B/cyc

        # Split by EFFECTIVE input rate, not nominal port width: an output-bound
        # array cannot consume at its port speed, so splitting by in_bw over-feeds
        # it and it finishes late.  With K1 != K2 the two arrays are almost always
        # bound differently, so this matters.
        tot_bw = sum(eff for _, _, _, eff in tiles)
        y = compute = 0.0
        for tin, tout, cost, eff in tiles:
            xi = x_bytes * eff / tot_bw         # now both arrays really do finish together
            n = xi / tin                        # number of tile ops
            y += n * tout
            compute = max(compute, n * cost)    # arrays run concurrently -> max

        launch = 100

    load, store = x_bytes / BW_DMA, y / BW_DMA
    return {"load": load, "compute": compute, "store": store, "out_bytes": y,
            "latency":   LAT + launch + load + compute + store,   # serial, one shot
            "pipelined": LAT + launch + max(load, compute, store)}  # double-buffered


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

def analytical(Br_SA1, Br_SA2, Bc, R,
               hw=Hardware(control="pipelined")):

    Br = Br_SA1 + Br_SA2
    G = N // K
    Q_offset = 0

    Digit_pairs = hw.digits**2

    # Your formulas assume pipelined command issue.
    # This helper receives TOTAL vector read + write traffic.
    def vector_cycles(total_bytes):
        return VCPU(
            in_bytes=total_bytes,
            out_bytes=0,
            hw=hw,
        ).launch_cycle()

    # Physical array dimensions and bandwidths come from the classes.
    sa1 = SA1(K=R, compute_model="mac")
    sa2 = SA2(K=R, compute_model="mac")

    # K/V loaded once for all G grouped Q heads.
    DMA_K = DMA(
        nbytes=2 * S * H, direction="read", hw=hw
    ).cycle()

    DMA_V = DMA(
        nbytes=2 * S * H, direction="read", hw=hw
    ).cycle()

    VCPU_KV = vector_cycles(10 * S * H + 16)
    KV_setup = DMA_K + DMA_V + VCPU_KV

    Q_head_BW = 0
    Q_head_MAC = 0

    for q0 in range(0, T, Br):
        r = min(Br, T - q0)
        r1 = min(r, Br_SA1)
        r2 = r - r1

        Row_tiles1 = ceil(r1 / sa1.physical_rows)
        Row_tiles2 = ceil(r2 / sa2.physical_rows)

        Padded_rows = (
            Row_tiles1 * sa1.physical_rows
            + Row_tiles2 * sa2.physical_rows
        )

        DMA_Q = DMA(
            nbytes=2 * r * H, direction="read", hw=hw
        ).cycle()

        DMA_O = DMA(
            nbytes=2 * r * H, direction="write", hw=hw
        ).cycle()

        VCPU_Q = vector_cycles(9 * r * H + 8 * r + 8)
        VCPU_O = vector_cycles(6 * r * H + 4 * r + 4)

        Q_head_BW += DMA_Q + DMA_O + VCPU_Q + VCPU_O
        Q_head_MAC += DMA_Q + DMA_O + VCPU_Q + VCPU_O

        for k0 in range(0, min(S, Q_offset + q0 + r), Bc):
            c = min(Bc, S - k0)

            Softmax = vector_cycles(17 * r * c + 28 * r)
            Update = vector_cycles(12 * r * H + 16 * r)

            Q_head_BW += Softmax + Update
            Q_head_MAC += Softmax + Update

            for product, columns, reduction in [
                ("QK", c, H),
                ("PV", H, c),
            ]:
                Chunks = ceil(reduction / R)

                # Preserve your fixed-depth padding.
                Padded_reduction = Chunks * R

                # Both arrays have 16 physical output columns.
                Physical_columns = sa1.output_bytes() // (
                    sa1.physical_rows * 2
                )
                Column_tiles = ceil(columns / Physical_columns)
                Padded_columns = Column_tiles * Physical_columns

                Jobs1 = Row_tiles1 * Column_tiles * Chunks * Digit_pairs
                Jobs2 = Row_tiles2 * Column_tiles * Chunks * Digit_pairs

                Raw_bytes = (
                    Jobs1 * sa1.output_bytes()
                    + Jobs2 * sa2.output_bytes()
                )

                Pack = vector_cycles(
                    (r + columns) * reduction
                    + hw.digits
                    * (Padded_rows + Padded_columns)
                    * Padded_reduction
                )

                Scale_bytes = 8 if product == "QK" else 4 * r + 4

                Recon = vector_cycles(
                    Raw_bytes
                    + 4 * r * columns
                    + Scale_bytes
                )

                # Preserve your first-command ordering.
                Start1 = 0 if Jobs1 > Jobs2 else hw.issue_interval
                Start2 = hw.issue_interval if Jobs1 > Jobs2 else 0

                End_BW = []
                End_MAC = []

                for sa, jobs, start in [
                    (sa1, Jobs1, Start1),
                    (sa2, Jobs2, Start2),
                ]:
                    if jobs == 0:
                        End_BW.append(0)
                        End_MAC.append(0)
                        continue

                    Input = sa.input_cycles()
                    Output = sa.output_cycles()
                    Compute = sa.compute_cycles()  # R in MAC model.

                    End_BW.append(
                        start
                        + hw.array_latency
                        + Input
                        + Output
                        + (jobs - 1) * max(Input, Output)
                    )

                    End_MAC.append(
                        start
                        + hw.array_latency
                        + Input
                        + Compute
                        + Output
                        + (jobs - 1) * max(Input, Compute, Output)
                    )

                Q_head_BW += Pack + Recon + max(End_BW)
                Q_head_MAC += Pack + Recon + max(End_MAC)

    First_Q_DMA = DMA(
        nbytes=2 * min(Br, T) * H,
        direction="read",
        hw=hw,
    ).cycle()

    Hidden_Q_DMA = min(
        First_Q_DMA,
        VCPU_KV - hw.vector_launch,
    )

    FA2_BW_cycles = KV_setup + G * Q_head_BW - Hidden_Q_DMA
    FA2_MAC_cycles = KV_setup + G * Q_head_MAC - Hidden_Q_DMA

    FA2_BW_ms = round(hw.time_us(FA2_BW_cycles) / 1000, 3)
    FA2_MAC_ms = round(hw.time_us(FA2_MAC_cycles) / 1000, 3)

    return FA2_BW_ms, FA2_MAC_ms