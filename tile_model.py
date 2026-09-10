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
