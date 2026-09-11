MB = 2 ** 20
GB = 2 ** 30

## Latency model
BW_DMA = 64                                     # DRAM -> DMA -> SRAM
BW_VCPU = 128                                   # SRAM <-> vector CPU
PORTS = {"sa16": [(16, 64, 8)],                 # (rows, in B/cyc, out B/cyc)
         "sa32": [(32, 96, 8)],
         "both": [(16, 64, 8), (32, 96, 8)]}


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