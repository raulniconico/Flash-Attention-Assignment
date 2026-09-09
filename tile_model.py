#!/usr/bin/env python3
"""FlashAttention tile-size evaluator for the supplied accelerator.

Run with Python 3.10+; no third-party packages are needed:
    python fa2_tile_model.py
    python fa2_tile_model.py --control pipelined --issue-interval 1
    python fa2_tile_model.py --array-model bandwidth
    python fa2_tile_model.py --br 128 --bc 256
    python fa2_tile_model.py --self-test

SCOPE
  One complete causal PREFILL attention head, N=2048, d=128 by default.
  Includes cold K/V reads and quantization, all Q reads, and all output writes.
  Excludes projections, RoPE, other transformer layers, and host transfers.
  K/V remain in SRAM. Q/K/V arrive as scaled 16-bit activations, are quantized
  to INT8, and the final output is encoded as scaled INT16. This introduces
  additional activation quantization; it is not strict A16 attention.

ARITHMETIC
  Default safe-int8 matches the most recent pseudocode: two signed base-16
  digits per INT8 operand, four digit-pair products, reduction chunks <=128.
  Each INT16 dot partial is safe because 128*15*15=28800. VCPU reconstructs
  the quantized product in INT32, then converts/scales it to FP32.
  native-int8 is a CONDITIONAL comparison only: one product, depth <=256.
  It requires a suitable documented accumulator/output conversion mechanism.
  Arbitrary full-range INT8 dots cannot be returned exactly in signed INT16.

SCHEDULE AND UNKNOWNS
  Phases execute serially: DMA, vector routine, matmul, vector routine, etc.
  The two arrays run concurrently. An array can compute its next tile while
  draining its previous output. No SRAM contention or bank conflicts.
  Full physical array operands/outputs are transferred (no assumed gating).
  All sweep dimensions are multiples of 16; there are no fractional tiles.
  The scheduler tries every whole-row division between the two arrays and
  uses an earliest-available greedy dispatch policy within each division.
  It is not a proof of the globally optimal hardware schedule.

  Array compute speed and command issue rate are not given in the assignment.
  Three array models expose that uncertainty:
    bandwidth: stage time = input transfer only (optimistic comparison).
    mac:       max(input transfer, K), assuming one MAC/PE/cycle, no fill.
    wavefront: max(input transfer, K+physical_rows+16-2), same MAC assumption
               plus conventional fill; one compute tile and one drain tile.
  wavefront is the default assumption, not a measured hardware fact.

  serialized control: wait for array acceptance, then a blocking 100-cycle
                      command, as in the earlier Gantt chart.
  pipelined control:  commands arrive after 100 cycles, but issue every I
                      cycles; timed issue avoids an array command queue.
  The specification gives 100-cycle latency, NOT I. The pipelined setting is
  a sensitivity scenario, not an assertion that the interface supports it.

  VCPU bandwidth is conservatively modeled as 128 B/cycle shared by reads
  and writes: ceil((read+write)/128), plus 300 cycles per routine. If the
  stated bidirectional bandwidth means 128 B/cycle independently each way,
  separate read/write ledgers and use their maximum instead.
  DMA links stream concurrently at min(64,64) B/cycle; do not add the two
  payload times. Per-transfer setup/first-byte latencies are included.

MEMORY AND CAUSAL WORK
  All K/V for one head are resident, including the original 16-bit copies.
  SRAM allocation is explicit, conservative, and does not alias buffers.
  Macroblocks above the diagonal are skipped. Within a diagonal block the
  arrays still compute the full rectangle before the vector causal mask.
  Br and Bc are SOFTWARE block sizes, not array dimensions. Bc can exceed
  256 because PV is split along its reduction dimension.

  Reported times are model predictions, not measured latency or full TTFT.
  With four Q heads sharing one KV head, cold KV setup can be amortized:
      group_cycles = kv_setup_cycles + 4 * query_work_cycles
  This assumes the same per-head query work and serial head execution.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache


def divup(a: int, b: int) -> int:
    return (a + b - 1) // b


@dataclass(frozen=True)
class Hardware:
    sram_bytes: int = 16 * 1024**2
    clock_hz: int = 1_000_000_000
    dma_bw: int = 64                 # both streaming links, B/cycle
    vector_bw: int = 128             # shared read + write, B/cycle
    dma_setup: int = 200
    dram_read_latency: int = 200
    dram_write_latency: int = 150
    vector_launch: int = 300
    array_latency: int = 100
    control: str = "serialized"
    issue_interval: int = 1          # only used for pipelined control
    array_model: str = "mac"
    arithmetic: str = "safe-int8"

    @property
    def digits(self) -> int:
        return 2 if self.arithmetic == "safe-int8" else 1

    @property
    def max_reduction(self) -> int:
        return 128 if self.arithmetic == "safe-int8" else 256


class SystolicArray:
    """One physical job. 'rows' and 'cols' count useful output elements.

    Partial jobs are padded to physical dimensions for traffic and timing.
    cycle() is the steady-state interval without CPU command overhead.
    latency() includes the final output drain for an isolated job.
    """
    physical_rows: int
    BW_in: int
    BW_out = 8

    def __init__(self, rows: int, K: int, cols: int = 16,
                 compute_model: str = "mac"):
        if not (1 <= rows <= self.physical_rows and 1 <= K <= 256
                and 1 <= cols <= 16):
            raise ValueError("invalid tile")
        if compute_model not in ("bandwidth", "mac", "wavefront"):
            raise ValueError("Unknown array model")
        self.rows, self.K, self.cols = rows, K, cols
        self.compute_model = compute_model

    def input_bytes(self) -> int:
        return (self.physical_rows + 16) * self.K  # two INT8 operands

    def output_bytes(self) -> int:
        return self.physical_rows * 16 * 2          # INT16 output

    def input_cycles(self) -> int:
        return divup(self.input_bytes(), self.BW_in)

    def output_cycles(self) -> int:
        return divup(self.output_bytes(), self.BW_out)

    def compute_cycles(self) -> int:
        if self.compute_model == "bandwidth":
            return 0
        if self.compute_model == "mac":
            return self.K
        return self.K + self.physical_rows + 16 - 2

    def stage_cycles(self) -> int:
        return max(self.input_cycles(), self.compute_cycles())

    def bandwidth_cycle(self) -> int:
        return max(self.input_cycles(), self.output_cycles())

    def cycle(self) -> int:
        return max(self.stage_cycles(), self.output_cycles())

    def latency(self) -> int:
        return self.stage_cycles() + self.output_cycles()

    def macs(self) -> int:
        return self.rows * self.cols * self.K

    def mac_rate(self) -> float:
        return self.macs() / self.cycle()


class SA1(SystolicArray):
    physical_rows = 16
    BW_in = 64


class SA2(SystolicArray):
    physical_rows = 32
    BW_in = 96


class VCPU:
    def __init__(self, in_bytes: int, out_bytes: int, hw: Hardware = Hardware()):
        self.in_bytes, self.out_bytes, self.hw = in_bytes, out_bytes, hw

    def cycle(self) -> int:
        return divup(self.in_bytes + self.out_bytes, self.hw.vector_bw)

    def launch_cycle(self) -> int:
        return self.hw.vector_launch + self.cycle()


class DMA:
    def __init__(self, nbytes: int, direction: str, hw: Hardware = Hardware()):
        if direction not in ("read", "write"):
            raise ValueError("DMA direction must be read or write")
        self.nbytes, self.direction, self.hw = nbytes, direction, hw

    def cycle(self) -> int:
        latency = (self.hw.dram_read_latency if self.direction == "read"
                   else self.hw.dram_write_latency)
        return self.hw.dma_setup + latency + divup(self.nbytes, self.hw.dma_bw)


@dataclass(frozen=True)
class ArrayCost:
    cycles: int
    jobs32: int
    jobs16: int
    rows32: int
    rows16: int
    raw_bytes: int


def schedule_arrays(m: int, n: int, reduction: int, tiles32: int,
                    hw: Hardware) -> ArrayCost:
    """Schedule physical dot jobs, including command arrival and final drains.

    Each engine has a compute stage and an output stage; it accepts a new
    compute tile once its previous tile enters the output stage. Distinct
    raw-output buffers preserve all partials until vector reconstruction.
    """
    tiles16 = (m - 32 * tiles32) // 16
    classes = (SA2, SA1)
    tile_counts = (tiles32, tiles16)
    depths = list(range(0, reduction, hw.max_reduction))
    sequences = [
        [min(hw.max_reduction, reduction - offset)
         for offset in depths
         for _ in range(hw.digits**2 * (n // 16) * count)]
        for count in tile_counts
    ]
    positions, accept, output_free = [0, 0], [0, 0], [0, 0]
    cpu_free = 0
    while any(positions[e] < len(sequences[e]) for e in (0, 1)):
        candidates = []
        for e in (0, 1):
            if positions[e] == len(sequences[e]):
                continue
            earliest_send = accept[e]
            if hw.control == "pipelined":
                earliest_send -= hw.array_latency
            send = max(cpu_free, earliest_send)
            remaining = len(sequences[e]) - positions[e]
            candidates.append((send, -remaining, e))
        send, _, e = min(candidates)
        job = classes[e](classes[e].physical_rows, sequences[e][positions[e]],
                         compute_model=hw.array_model)
        start = send + hw.array_latency
        assert start >= accept[e]
        issue = (hw.array_latency if hw.control == "serialized"
                 else hw.issue_interval)
        cpu_free = send + issue
        out_start = max(start + job.stage_cycles(), output_free[e])
        accept[e] = out_start
        output_free[e] = out_start + job.output_cycles()
        positions[e] += 1
    return ArrayCost(max(cpu_free, *output_free), len(sequences[0]),
                     len(sequences[1]), tiles32 * 32, tiles16 * 16,
                     2 * m * n * hw.digits**2 * len(depths))


@lru_cache(maxsize=None)
def best_array_cost(m: int, n: int, reduction: int, hw: Hardware) -> ArrayCost:
    """Best whole-row allocation under the stated greedy dispatch policy."""
    choices = [schedule_arrays(m, n, reduction, t, hw)
               for t in range(m // 32 + 1)]
    return min(choices, key=lambda cost: cost.cycles)


class Ledger:
    def __init__(self, hw: Hardware):
        self.hw = hw
        self.cycles = {"dma": 0, "vector": 0, "arrays": 0}
        self.vector_bytes = 0
        self.dram_read_bytes = 0
        self.dram_write_bytes = 0
        self.jobs32 = 0
        self.jobs16 = 0

    @property
    def total(self) -> int:
        return sum(self.cycles.values())

    def dma(self, nbytes: int, direction: str) -> None:
        self.cycles["dma"] += DMA(nbytes, direction, self.hw).cycle()
        if direction == "read":
            self.dram_read_bytes += nbytes
        else:
            self.dram_write_bytes += nbytes

    def vector(self, total_bytes: int) -> None:
        # The stage-specific ledgers below already sum reads and writes.
        self.cycles["vector"] += VCPU(total_bytes, 0, self.hw).launch_cycle()
        self.vector_bytes += total_bytes

    def matmul(self, m: int, n: int, reduction: int, score: bool) -> None:
        if self.hw.digits == 2:
            # Read INT8 operands once, write two INT8 signed digit planes.
            self.vector(3 * (m + n) * reduction)
        cost = best_array_cost(m, n, reduction, self.hw)
        self.cycles["arrays"] += cost.cycles
        self.jobs32 += cost.jobs32
        self.jobs16 += cost.jobs16
        scales = 8 if score else 4 * m + 4
        # Read every INT16 partial, reduce/reconstruct in registers,
        # write FP32 result, read scalar or rowwise quantization scales.
        self.vector(cost.raw_bytes + 4 * m * n + scales)


def sram_allocation(N: int, d: int, br: int, bc: int,
                    hw: Hardware) -> dict[str, int]:
    """Fixed buffers, no aliasing, no double buffering. Bytes, not bits."""
    m, c = min(br, N), min(bc, N)
    digits = hw.digits
    chunks_qk = divup(d, hw.max_reduction)
    chunks_pv = divup(c, hw.max_reduction)
    return {
        "K16 + V16": 4 * N * d,
        "K8 + packed V8 transpose": 2 * N * d,
        "Q16 + Q8": 3 * m * d,
        "FP32 output numerator U": 4 * m * d,
        "FP32 PV contribution": 4 * m * d,
        "INT16 final output": 2 * m * d,
        "FP32 scores + FP32 P + INT8 P": 9 * m * c,
        "row statistics and scales": 32 * m + 64,
        "reusable operand digit planes": (
            digits * max((m + c) * d, (m + d) * c) if digits == 2 else 0),
        "reusable INT16 raw partials": (
            2 * digits**2 * max(m * c * chunks_qk, m * d * chunks_pv)),
    }


def evaluate(N: int, d: int, br: int, bc: int,
             hw: Hardware = Hardware()) -> dict:
    if any(x <= 0 or x % 16 for x in (N, d, br, bc)):
        raise ValueError("This evaluator requires N, d, Br and Bc multiples of 16")
    allocation = sram_allocation(N, d, br, bc, hw)
    used = sum(allocation.values())
    if used > hw.sram_bytes:
        return {"br": br, "bc": bc, "fits": False, "sram_bytes": used}
    L = Ledger(hw)
    L.dma(2 * N * d, "read")                        # K16
    L.dma(2 * N * d, "read")                        # V16
    L.vector(10 * N * d + 16)                       # two-pass KV quantization
    kv_setup_cycles = L.total
    blocks = 0
    rectangles = 0
    for q0 in range(0, N, br):
        m = min(br, N - q0)
        L.dma(2 * m * d, "read")                    # Q16
        L.vector(9 * m * d + 8 * m + 8)             # quantize Q, initialize U/l/m
        for k0 in range(0, q0 + m, bc):
            c = min(bc, N - k0)
            L.matmul(m, c, d, score=True)            # QK^T -> FP32 scores
            # Three passes: max; exp/sum/row scale; INT8 P encoding.
            # Includes masking in the first two passes, computed for free.
            L.vector(17 * m * c + 28 * m)
            L.matmul(m, d, c, score=False)           # P V -> FP32 contribution
            L.vector(12 * m * d + 16 * m)            # alpha U + contribution; l
            blocks += 1
            rectangles += m * c
        L.vector(6 * m * d + 4 * m + 4)             # normalize, encode output
        L.dma(2 * m * d, "write")
    valid_pairs = N * (N + 1) // 2
    return {
        "br": br, "bc": bc, "fits": True, "sram_bytes": used,
        "cycles": L.total, "ms": L.total / hw.clock_hz * 1000,
        "stage_cycles": L.cycles, "kv_setup_cycles": kv_setup_cycles,
        "query_work_cycles": L.total - kv_setup_cycles,
        "blocks": blocks, "rectangle_pairs": rectangles,
        "causal_utilization": valid_pairs / rectangles,
        "logical_macs": 2 * d * rectangles,
        "useful_macs": 2 * d * valid_pairs,
        "array_jobs32": L.jobs32, "array_jobs16": L.jobs16,
        "vector_bytes": L.vector_bytes,
        "dram_read_bytes": L.dram_read_bytes,
        "dram_write_bytes": L.dram_write_bytes,
        "allocation": allocation,
    }


def self_test() -> None:
    # Independent hand calculations for physical bandwidth/compute/latency.
    a, b = SA1(16, 128), SA2(32, 128)
    assert (a.input_cycles(), a.output_cycles()) == (64, 64)
    assert (b.input_cycles(), b.output_cycles()) == (64, 128)
    assert (a.cycle(), b.cycle()) == (158, 174)
    assert (a.latency(), b.latency()) == (222, 302)
    assert DMA(524288, "read").cycle() == 8592
    assert DMA(32768, "write").cycle() == 862
    assert VCPU(128, 128).cycle() == 2
    # A single safe INT8 tile needs four physical jobs; no second array.
    hw = Hardware()
    cost = schedule_arrays(16, 16, 128, 0, hw)
    assert cost.jobs16 == 4 and cost.jobs32 == 0
    assert cost.cycles == 4 * (100 + 158) + 64
    long_dot = schedule_arrays(16, 16, 256, 0, hw)
    assert long_dot.jobs16 == 8
    assert long_dot.raw_bytes == 2 * 16 * 16 * 8
    # Validate digit decomposition and the worst-case signed INT16 bound.
    for x in range(-127, 128):
        sign = -1 if x < 0 else 1
        low, high = sign * (abs(x) % 16), sign * (abs(x) // 16)
        assert x == low + 16 * high
        assert abs(low) <= 15 and abs(high) <= 7
    assert 128 * 15**2 <= 32767
    # Causal block accounting, including a last short Q/K tile.
    r = evaluate(80, 32, 32, 48, hw)
    assert r["blocks"] == 5
    assert r["rectangle_pairs"] == 32*48 + 32*80 + 16*80
    assert r["dram_read_bytes"] == 6 * 80 * 32
    assert r["dram_write_bytes"] == 2 * 80 * 32
    # For this tiny problem, exactly sum the stage ledgers by hand.
    r = evaluate(16, 16, 16, 16, hw)
    assert r["stage_cycles"] == {"dma": 1582, "vector": 2892, "arrays": 1296}
    assert r["cycles"] == 5770
    # Same 100-cycle arrival latency, faster issue, and a 64-cycle drain.
    pipelined = Hardware(control="pipelined", issue_interval=1)
    cost = schedule_arrays(16, 16, 16, 0, pipelined)
    assert cost.cycles == 100 + 46 + 4 * 64
    print("Self-test passed: physical timing, DMA, arithmetic safety, causal work, full ledger.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--N", type=int, default=2048)
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--br", type=int, help="Evaluate a single query block size")
    p.add_argument("--bc", type=int, help="Evaluate a single key block size")
    p.add_argument("--control", choices=("serialized", "pipelined"), default="serialized")
    p.add_argument("--issue-interval", type=int, default=1)
    p.add_argument("--array-model", choices=("bandwidth", "mac", "wavefront"),
                   default="wavefront")
    p.add_argument("--arithmetic", choices=("safe-int8", "native-int8"), default="safe-int8")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    if args.self_test:
        self_test()
        return
    if args.issue_interval < 1:
        p.error("--issue-interval must be positive")
    if (args.br is None) != (args.bc is None):
        p.error("Supply both --br and --bc, or neither for a sweep")
    hw = Hardware(control=args.control, issue_interval=args.issue_interval,
                  array_model=args.array_model, arithmetic=args.arithmetic)
    brs = [args.br] if args.br else [32, 64, 96, 128, 192, 256, 384, 512]
    bcs = [args.bc] if args.bc else [32, 64, 128, 256, 512]
    results = [evaluate(args.N, args.d, br, bc, hw) for br in brs for bc in bcs]
    valid = sorted((r for r in results if r["fits"]), key=lambda r: r["cycles"])
    print(f"One causal prefill head: N={args.N}, d={args.d}; includes cold K/V setup")
    print(f"Assumptions: {hw.arithmetic}, {hw.array_model}, {hw.control} control")
    if hw.control == "pipelined":
        print(f"Hypothetical command issue interval: {hw.issue_interval} cycles; latency stays 100")
    if hw.arithmetic == "native-int8":
        print("CONDITIONAL: native full-range INT8 needs a defined overflow-safe output mechanism.")
    print("Ranking is within this candidate grid and schedule, not a measured hardware optimum.\n")
    print(" Br   Bc    SRAM MiB    time ms    Q/K blocks    causal use    array jobs")
    for r in valid:
        print(f'{r["br"]:4} {r["bc"]:4} {r["sram_bytes"]/1024**2:11.3f}'
              f' {r["ms"]:10.4f} {r["blocks"]:13}'
              f' {100*r["causal_utilization"]:11.1f}%'
              f' {r["array_jobs32"]+r["array_jobs16"]:13,}')
    for r in results:
        if not r["fits"]:
            print(f'{r["br"]:4} {r["bc"]:4} {r["sram_bytes"]/1024**2:11.3f}  DOES NOT FIT')
    if not valid:
        return
    winner = valid[0]
    print(f'\nBest tested: Br={winner["br"]}, Bc={winner["bc"]}, {winner["ms"]:.4f} ms')
    print("Serial stage totals (command/launch cost already included):")
    for name, cycles in winner["stage_cycles"].items():
        print(f"  {name:8}: {cycles:,} cycles ({100*cycles/winner['cycles']:.1f}%)")
    print(f"  cold KV setup: {winner['kv_setup_cycles']:,} cycles")
    print("SRAM allocation:")
    for name, nbytes in winner["allocation"].items():
        print(f"  {name:39} {nbytes:11,} B")
    m, c = min(winner["br"], args.N), min(winner["bc"], args.N)
    qk = best_array_cost(m, c, args.d, hw)
    pv = best_array_cost(m, args.d, c, hw)
    print("Row assignment for a full block (may differ on tails):")
    print(f"  QK: SA32 gets {qk.rows32} rows; SA16 gets {qk.rows16} rows")
    print(f"  PV: SA32 gets {pv.rows32} rows; SA16 gets {pv.rows16} rows")


if __name__ == "__main__":
    main()
