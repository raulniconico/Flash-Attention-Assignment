from helper import *

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

byte_list = [16 * MB]
device_list = ["sa16", "sa32", "both"]
K1_list = [32, 64, 128, 256]
K2_list = [32, 64, 128, 256]

print(f"{'size':>8} {'device':>7} {'K1/K2':>9} {'transfer':>12} {'compute':>12} "
      f"{'total':>13} {'transfer%':>10}")
print("-" * 78)

for x in byte_list:
    for unit in device_list:
        # both arrays -> sweep a depth for each; a single array has only K1
        pairs = ([(k1, k2) for k1 in K1_list for k2 in K2_list] if unit == "both"
                 else [(k1, None) for k1 in K1_list])
        for K1, K2 in pairs:
            r = kernel_time(x, unit, K1, K2)
            transfer = r["load"] + r["store"]      # DRAM->SRAM  +  SRAM->DRAM
            compute = r["compute"]
            total = r["latency"]                   # transfer + compute + LAT
            label = f"{K1}/{K2}" if K2 is not None else f"{K1}"
            print(f"{x // MB:>6}MB {unit:>7} {label:>9} {transfer:>12,.0f} {compute:>12,.0f} "
                  f"{total:>13,.0f} {100 * transfer / total:>9.2f}%")
        print("-" * 78)

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