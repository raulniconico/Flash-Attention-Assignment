#!/usr/bin/env python3
"""
perf_model_annotated.py
=============================================================================
A plain-language rewrite of C2/perf_model.py.

Identical code, identical output -- but every function explains what it is
computing and why, instead of assuming you already know. Companion to
flash_attention_annotated.py and report_annotated.md.


=============================================================================
WHAT IS THIS FILE FOR?
=============================================================================

It is the CALCULATOR BEHIND THE REPORT.

Every number in report.md -- 157 s, 189 ms, -0.34, 0.82 GB/s -- comes out of
this file. Nothing is estimated by hand. Run it and the report reprints
itself:

    python3 perf_model_annotated.py                 # print everything
    python3 perf_model_annotated.py --json out.json # dump raw results
    python3 perf_model_annotated.py --plot          # write roofline.png

It answers four separate questions, in this order:

    Problem 2  ->  which parameter is the bottleneck?   (section 5, elasticity)
    Problem 3  ->  how fast is it?                      (sections 3-4)
    Problem 4  ->  what should we change?               (section F of main)
    Problem 5  ->  how does it scale to 4 chips?        (section 6)

The kernel design itself is the other file (flash_attention_pseudocode.py).
This one never describes HOW to compute attention -- only how LONG it takes.


=============================================================================
THE MODELLING PHILOSOPHY -- four ideas, and that is the whole model
=============================================================================

IDEA 1: EVERY ENGINE IS A PIPE.
    Forget what a component *does*. Ask only how many bytes per cycle it can
    move. At 1 GHz, 1 cycle = 1 ns, so:

        time on an engine = bytes moved / bandwidth

    That is it. The vector CPU has "infinite compute" per the assignment, so
    its time is literally bytes/128. The DMA is bytes/64. Even the systolic
    arrays are modelled this way (idea 2).

IDEA 2: THE ARRAYS ARE THEIR PORTS, NOT THEIR MULTIPLIERS.
    A tile op streams two int8 operands IN and one int16 result OUT. The
    array is pipelined -- the next op's inputs flow in while this op's
    outputs flow out -- so it costs:

        max(input bytes / input bw,  output bytes / output bw)

    NOT the sum. This is `tile_cycles()` below, and it is the single most
    consequential line in the file: it is what makes the 32x16 array's
    8 B/cycle output port the answer to Problem 2.

    Sanity check that the assignment's constants were chosen deliberately:
    at K = 256 both arrays land on exactly 128 cycles per tile. That is not
    a coincidence.

IDEA 3: CONCURRENT ENGINES COMBINE WITH max(), NOT sum().
    Inside a phase, the DMA, the two arrays and the vector CPU all run at
    once, because the kernels are double-buffered. So the slowest one hides
    the others:

        phase time = max(array, dram, vector) + fill

    This is why every result table has four numeric columns and a "bound by"
    column. The model does NOT decide in advance what the bottleneck is --
    it computes all of them and reports which won. That is what makes the
    bottleneck claims in the report evidence rather than assertion.

IDEA 4: FIXED LATENCIES ARE PAID ONCE PER PHASE, NOT PER OPERATION.
    The control CPU issues commands into queues ahead of time (assumption
    A7), so the 200/300/100-cycle latencies are a one-off pipeline-fill cost.

    This assumption is worth a lot: `sa_cmd_queue = False` below turns it off
    and peak throughput falls from 1536 to 862 MAC/cycle. The report quotes
    that number; this flag is where it comes from.


=============================================================================
HOW TO READ THIS FILE
=============================================================================

    section 1   the chip, as ~10 bandwidth numbers
    section 2   the model and the workload (Llama 3.1 8B, batch 16)
    section 3   how long one systolic-array tile op takes   <- the core
    section 4   how long each phase of a layer takes
    section 5   elasticity: wiggle a parameter, measure the effect
    section 6   four accelerators on one host
    section 7   printing, and the roofline plot

Sections 3 and 4 are the model. Sections 5-6 just call them repeatedly with
modified inputs -- which is the point of building it this way: once a phase
is a pure function of an Arch, asking "what if this wire were wider?" is a
function call, not a rewrite.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict, replace


# ------------------------------------------------------------------------------------
# 1. The chip
#
#    This is Figure 1's parameter table and nothing else. Every "512 bit/cycle"
#    becomes 64 bytes/cycle. At 1 GHz that is also 64 GB/s.
#
#    Note how small the two `*_out` values are: 8 B/cycle, EIGHT TIMES narrower
#    than the input port feeding the same array. That asymmetry is the answer
#    to Problem 2, and it is visible right here in the constants.
# ------------------------------------------------------------------------------------
@dataclass
class Arch:
    f_hz: float = 1.0e9
    bw_dram_dma: float = 512 / 8       # 64  B/cycle  DRAM  <-> DMA
    bw_dma_sram: float = 512 / 8       # 64  B/cycle  DMA   <-> SRAM   (in SERIES with the above)
    bw_sram_vcpu: float = 1024 / 8     # 128 B/cycle  SRAM  <-> vector CPU  (widest port on the chip)
    sa16_in: float = 512 / 8           # 64  B/cycle  SRAM  -> 16x16 array
    sa16_out: float = 64 / 8           # 8   B/cycle  16x16 array -> SRAM     <-- narrow
    sa32_in: float = 768 / 8           # 96  B/cycle  SRAM  -> 32x16 array
    sa32_out: float = 64 / 8           # 8   B/cycle  32x16 array -> SRAM     <-- narrow
    lat_ctrl_dma: float = 200          # cycles: control CPU -> DMA
    lat_ctrl_vcpu: float = 300         # cycles: control CPU -> vector CPU
    lat_ctrl_sa: float = 100           # cycles: control CPU -> systolic array
    lat_dram_rd: float = 200
    lat_dram_wr: float = 150
    sram_bytes: int = 16 << 20         # 16 MiB
    dram_bytes: int = 64 << 30         # 64 GiB
    kmax: int = 256                    # max contraction depth of one tile op
    sa_in_bytes: int = 1               # arrays take int8 in
    sa_out_bytes: int = 2              # arrays emit int16 out

    # ---- a switch, not a constant: assumption A7 ----
    # True  = tile commands are queued, so the 100-cycle control latency is paid
    #         once per burst. This is what the kernel design achieves.
    # False = pay it on EVERY tile op. Peak drops 1536 -> 862 MAC/cycle.
    # Flipping this is how section D of main() quantifies the assumption.
    sa_cmd_queue: bool = True

    @property
    def bw_dma(self) -> float:
        """Effective DRAM -> SRAM streaming bandwidth.

        min(), not sum(): the DMA engine sits BETWEEN two 64 B/cycle links and
        data must cross both. They are in series, so the path is only as wide
        as its narrower half.

        This one-line detail is why Problem 2 reports -0.55 for either link
        alone but -1.01 for the two together, and why Improvement 1 in
        Problem 4 insists both must be widened."""
        return min(self.bw_dram_dma, self.bw_dma_sram)

    def sec(self, cycles: float) -> float:
        """cycles -> seconds. At 1 GHz this is just a division by 1e9."""
        return cycles / self.f_hz


# ------------------------------------------------------------------------------------
# 2. The model being run, and the job being asked of it
# ------------------------------------------------------------------------------------
@dataclass
class LlamaModel:
    """Llama 3.1 8B. Every field is a published architecture constant (A15)."""
    name: str = "Llama 3.1 8B"
    n_layers: int = 32
    d_model: int = 4096
    n_heads: int = 32          # query heads
    n_kv_heads: int = 8        # KV heads -- fewer, because of GQA
    d_head: int = 128          # <- note: this is the K of Q.K^T, and K=128 is
                               #    exactly where the 32x16 array is output-bound
    d_ff: int = 14336
    vocab: int = 128256

    @property
    def d_q(self) -> int:
        return self.n_heads * self.d_head

    @property
    def d_kv(self) -> int:
        return self.n_kv_heads * self.d_head

    @property
    def heads_per_group(self) -> int:
        """4. Grouped-Query Attention: 4 query heads share one K/V set.

        This is why prefill can load K/V once and reuse it 4 times, and why
        the tensor-parallel plan in Problem 5 shards cleanly by 4."""
        return self.n_heads // self.n_kv_heads

    def layer_gemms(self):
        """The four matrix multiplies in one transformer layer, as (name, K, N),
        acting on activations of shape [tokens x K].

        Weights are stored TRANSPOSED ([N x K], K contiguous) so that every
        operand handed to a systolic array is already a row of K contiguous
        8-bit values -- the only shape the arrays accept (A11)."""
        return [
            ("qkv_proj", self.d_model, self.d_q + 2 * self.d_kv),
            ("o_proj", self.d_q, self.d_model),
            ("gate_up_proj", self.d_model, 2 * self.d_ff),
            ("down_proj", self.d_ff, self.d_model),
        ]

    @property
    def linear_params_per_layer(self) -> int:
        return sum(K * N for _, K, N in self.layer_gemms())

    @property
    def linear_params(self) -> int:
        return self.n_layers * self.linear_params_per_layer

    @property
    def lm_head_params(self) -> int:
        return self.d_model * self.vocab

    @property
    def embed_params(self) -> int:
        return self.d_model * self.vocab

    @property
    def total_params(self) -> int:
        return self.linear_params + self.lm_head_params + self.embed_params

    def kv_bytes_per_token(self, kv_bytes: float) -> float:
        """How much KV cache ONE token costs, across the whole model.

            32 layers x 2 tensors (K and V) x 1024 dims x 2 bytes = 128 KiB

        Small per token -- but multiply by 16 sequences x ~2176 context and it
        becomes 4.56 GB that must be re-read on EVERY decode step. That is 38%
        of the decode step time."""
        return self.n_layers * 2 * self.d_kv * kv_bytes


@dataclass
class Workload:
    """The job: 16 conversations, 2048-token prompts, 256 generated tokens."""
    batch: int = 16
    prompt: int = 2048
    gen: int = 256
    w_bytes: float = 1.0       # weight element in DRAM: int8
                               # (Improvement 2 sets this to 0.5 for int4)
    act_bytes: int = 2         # activation element: int16
    kv_bytes: float = 2.0      # KV cache element: int16, per the spec
                               # (Improvement 2 sets this to 1.0 for int8)
    acc_bytes: int = 4         # vector-CPU accumulator: fp32
    br: int = 128              # flash-attention query block (4 chunks of 32 rows)
    bc: int = 256              # flash-attention key block  (= Kmax of the arrays)
    slab_bytes: int = 8 << 20  # weight slab kept resident in SRAM during prefill

    # Note that w_bytes and kv_bytes are FIELDS, not constants. That is what
    # lets Problem 4 evaluate "int4 weights + int8 KV" by changing two numbers
    # and re-running, with no other edits anywhere.


# ------------------------------------------------------------------------------------
# 3. The systolic-array tile model
#
#    THIS IS THE HEART OF THE FILE. Three lines of arithmetic that produce the
#    array table in report section 0.2(b) -- rather than restating it.
# ------------------------------------------------------------------------------------
def tile_cycles(a: Arch, rows_a: int, K: int) -> float:
    """How many cycles one tile op takes on the (rows_a x 16) array.

    A tile op streams in `rows_a` rows of A and 16 rows of B, each K int8
    values, and streams out a rows_a x 16 tile of int16 results.

        t_in  = (rows_a + 16) * K * 1 byte  /  input bandwidth
        t_out = rows_a * 16 * 2 bytes       /  output bandwidth
        cost  = max(t_in, t_out)

    WHY max AND NOT sum: the array is pipelined. Tile n+1's inputs stream in
    while tile n's outputs stream out. So the slower of the two directions
    sets the pace and the other is free.

    NOW LOOK AT WHAT t_out DOES NOT DEPEND ON: K.
    The output tile is always rows_a x 16 regardless of contraction depth. So
    halving K halves the input time and leaves the output time untouched:

        32x16, K=256:  in 128 cyc, out 128 cyc  -> 128, balanced, 1024 MAC/cyc
        32x16, K=128:  in  64 cyc, out 128 cyc  -> 128, OUTPUT-BOUND, 512 MAC/cyc

    The head dimension of the model is 128. So on Q.K^T, the big expensive
    32x16 array delivers exactly the same throughput as the small one, because
    of a wire 8x narrower than the one feeding it.

    That is the answer to Problem 2, and it falls out of these three lines."""
    in_bw, out_bw = (a.sa16_in, a.sa16_out) if rows_a == 16 else (a.sa32_in, a.sa32_out)
    t_in = (rows_a + 16) * K * a.sa_in_bytes / in_bw
    t_out = rows_a * 16 * a.sa_out_bytes / out_bw
    t = max(t_in, t_out)
    if not a.sa_cmd_queue:
        # Assumption A7 switched off: pay the control latency on every single
        # op instead of once per burst. A 100-cycle wait per 128-cycle tile.
        t += a.lat_ctrl_sa
    return t


def mac_rate(a: Arch, rows_a: int, K: int) -> float:
    """Useful multiply-accumulates per cycle for one array at depth K.

    A tile op performs rows_a * 16 * K MACs and costs tile_cycles(). Divide.
    This is the number that appears in the report's array table."""
    return rows_a * 16 * K / tile_cycles(a, rows_a, K)


def peak_mac_rate(a: Arch) -> float:
    """Both arrays at their best depth (K = 256): 512 + 1024 = 1536 MAC/cycle.

    = 3.07 TOPS. And divided by the 64 B/cycle DRAM path it gives 24 MAC per
    DRAM byte -- the RIDGE POINT, the single number the entire report turns on.
    Above 24: compute-bound. Below: memory-bound."""
    return mac_rate(a, 16, a.kmax) + mac_rate(a, 32, a.kmax)


# ------------------------------------------------------------------------------------
# 4. The phase model
#
#    A "phase" is one step of a transformer layer: a matmul, or attention, or
#    the element-wise ops. The model computes each phase's time on EVERY engine
#    and then takes the max.
# ------------------------------------------------------------------------------------
@dataclass
class Phase:
    """One phase's timing, engine by engine.

    Keeping all three engine times (not just the winner) is deliberate: it is
    what lets the report print "537 / 21 / 264 -> arrays" and show its work
    instead of just asserting a bottleneck."""
    name: str
    cycles: float          # the actual answer: max(engines) + fill
    t_array: float = 0.0   # what it would take if only the arrays mattered
    t_dram: float = 0.0    # ... only the DMA path
    t_vcpu: float = 0.0    # ... only the vector CPU
    macs: float = 0.0
    dram_bytes: float = 0.0
    vcpu_bytes: float = 0.0

    @property
    def bound(self) -> str:
        """Which engine actually won. This is the "bound by" column of every
        table in the report -- computed, never assumed."""
        d = {"array": self.t_array, "dram": self.t_dram, "vcpu": self.t_vcpu}
        return max(d, key=d.get)


def fill_latency(a: Arch) -> float:
    """Pipeline-fill cost, paid ONCE per phase (assumption A7).

    Because the control CPU queues commands ahead of time, these four
    latencies are a startup cost, not a per-operation cost. 800 cycles = 0.8 us
    against phases lasting milliseconds -- which is exactly why the control
    latencies show elasticity 0.00 in Problem 2. That zero is an achievement
    of the kernel design, not a property of the hardware."""
    return a.lat_ctrl_dma + a.lat_dram_rd + a.lat_ctrl_sa + a.lat_ctrl_vcpu


def gemm(a: Arch, wl: Workload, name: str, M: int, K: int, N: int,
         act_in_sram: bool = False, extra_dram_bytes: float = 0.0) -> Phase:
    """Time one matrix multiply  Y[M x N] = X[M x K] . W[K x N].

    M = number of tokens in flight. THIS SINGLE ARGUMENT is what flips the
    machine between its two regimes:

        prefill: M = 32768  -> each weight byte serves 32768 tokens -> compute-bound
        decode:  M = 16     -> each weight byte serves 16 tokens    -> memory-bound

    Same function, same weights, opposite answer. That is the whole report.

    `act_in_sram=True` marks the decode case, where the [16 x K] activation
    tile is tiny and stays resident, so it costs no DRAM traffic at all.

    The three engine costs:

    ARRAY -- both arrays chew disjoint output tiles; each op streams a 16-row
        activation tile and a 16/32-row weight tile through the input port.

    DRAM -- weights once, always. Plus, in PREFILL only, the activation
        traffic: the [M x K] matrix is far too big for SRAM, so it is
        re-read once per weight slab that fits (A9). With M = 32768 that
        re-reading is real traffic -- but prefill has DRAM to spare.

    VECTOR -- three jobs, all forced by hardware limitations rather than by
        the math:
          (i)   requantise the 16-bit input to int8 (arrays only eat int8)
          (ii)  accumulate each 256-deep chunk's int16 partial into fp32,
                because the arrays CANNOT accumulate into an existing C tile
                (A4). This is the big one -- it is most of why the vector CPU
                sits at ~49% during prefill.
          (iii) read the accumulator, write the 16-bit output
    """
    M_eff = math.ceil(M / 16) * 16                 # the B operand is always a full 16-row tile
                                                   # (decode's M=16 is exactly one tile; nothing wasted)
    Kc = min(K, a.kmax)                            # contraction is chopped into <=256-deep chunks
    n_kc = math.ceil(K / Kc)                       # how many chunks -> how many partial sums to fold
    macs = M * K * N                               # useful work
    macs_issued = M_eff * K * N                    # work actually issued (padding included)
    rate = mac_rate(a, 16, Kc) + mac_rate(a, 32, Kc)
    t_array = macs_issued / rate

    w_bytes = K * N * wl.w_bytes
    if act_in_sram:
        act_traffic = 0.0                          # decode: activations never leave SRAM
    else:
        n_slabs = math.ceil(w_bytes / wl.slab_bytes)
        act_traffic = (M * K * wl.act_bytes              # read the 16-bit input once
                       + M * K * a.sa_in_bytes           # write the int8 copy
                       + n_slabs * M * K * a.sa_in_bytes # re-read that copy once per weight slab
                       + M * N * wl.act_bytes)           # write the 16-bit output
    dram_bytes = w_bytes + act_traffic + extra_dram_bytes
    t_dram = dram_bytes / a.bw_dma

    requant = M * K * (wl.act_bytes + a.sa_in_bytes)
    partial = n_kc * (M_eff * N * a.sa_out_bytes + 2 * M * N * wl.acc_bytes)
    final = M * N * (wl.acc_bytes + wl.act_bytes)
    vcpu_bytes = requant + partial + final
    t_vcpu = vcpu_bytes / a.bw_sram_vcpu

    # IDEA 3: the engines run concurrently, so the slowest one is the answer.
    cycles = max(t_array, t_dram, t_vcpu) + fill_latency(a)
    return Phase(name, cycles, t_array, t_dram, t_vcpu, macs, dram_bytes, vcpu_bytes)


def causal_blocks(S: int, br: int, bc: int) -> int:
    """Count the (query block, key block) pairs that are NOT entirely in the future.

    A word cannot attend to words that come after it, so roughly the top-right
    half of the score matrix is never computed. Query block i covers queries up
    to q_last = (i+1)*br - 1; key block j is needed only if it starts at or
    before that.

    For S=2048, br=128, bc=256 this returns 72 out of 128 blocks -- the 0.5625
    causal factor quoted throughout the report."""
    n = 0
    for i in range(S // br):
        q_last = (i + 1) * br - 1
        n += sum(1 for j in range(S // bc) if j * bc <= q_last)
    return n


def split_two_arrays(w_qk: float, w_pv: float,
                     r16_qk: float, r32_qk: float, r16_pv: float, r32_pv: float):
    """How should Q.K^T and P.V be divided between two UNEQUAL arrays?

    The situation (from tile_cycles above):
        on Q.K^T (K=128): both arrays give 512 MAC/cyc -- a TIE
        on P.V   (K=256): the 32x16 gives 1024, the 16x16 gives 512 -- 2x

    So the 32x16 array has no advantage on scores but a big one on outputs.
    "Half each" is therefore wrong. The right split gives the big array the
    work only it is good at.

    This is a MAKESPAN problem: two workers, two job types, different speeds
    each. Minimise max(T16, T32) -- the finish time of whoever ends last,
    because the other one just sits idle waiting.

    x = fraction of Q.K^T given to the 32x16
    y = fraction of P.V   given to the 32x16

    For a fixed y, T16 falls and T32 rises as x grows, so the optimum is where
    the two lines cross (clipped to [0,1]). Sweep y over 1000 points, solve for
    the crossing at each, keep the best.

    Answer: x = 0.25, y = 1.0. Which is to say:
        16x16 -> 75% of Q.K^T
        32x16 -> 25% of Q.K^T + ALL of P.V
    giving 1365 MAC/cycle sustained = 89% of the 1536 peak.

    (Splitting work proportionally to array speed instead would give 1229,
    because the 32x16 would idle whenever only Q.K^T work was available. The
    report cites this function as confirmation of the algebra in section 1.A.)

    Returns (cycles, x, y).
    """
    a_, b_ = w_qk / r16_qk, w_pv / r16_pv      # cost of doing 100% of each job on the 16x16
    c_, d_ = w_qk / r32_qk, w_pv / r32_pv      # ... and on the 32x16
    best = (math.inf, 0.0, 0.0)
    steps = 1000
    for k in range(steps + 1):
        y = k / steps
        # Solve T16(x) = T32(x) for this y:
        #     (1-x)a + (1-y)b  =  xc + yd
        x = (a_ + (1 - y) * b_ - y * d_) / (a_ + c_)
        x = min(1.0, max(0.0, x))              # a crossing outside [0,1] means one
                                               # array should simply take everything
        t16 = (1 - x) * a_ + (1 - y) * b_
        t32 = x * c_ + y * d_
        t = max(t16, t32)                      # the makespan
        if t < best[0]:
            best = (t, x, y)
    return best


def attention_prefill_layer(a: Arch, wl: Workload, m: LlamaModel) -> Phase:
    """Time the prefill flash-attention kernel for one layer (Part A of the pseudocode).

    Per (sequence, KV group): K/V are loaded once and shared by the 4 query
    heads of the group (GQA). Blocks of 128 queries x 256 keys stream through
    a 3-stage pipeline:

        [Q.K^T on the arrays] -> [online softmax on the vector CPU] -> [P.V on the arrays]

    Note what is NOT modelled: the one-block software pipelining, the double
    buffers, the 3/1 array split of the sub-tiles. Those are correctness and
    scheduling details that live in the pseudocode. Here they are assumed to
    work, and their effect shows up as a single number -- the makespan from
    split_two_arrays(). The model asks "given a perfect schedule, what is the
    floor?", and the pseudocode's job is to reach it."""
    S, D = wl.prompt, m.d_head
    hpg = m.heads_per_group
    nblk = causal_blocks(S, wl.br, wl.bc)          # 72 of 128

    # Both matmuls have the SAME MAC count: Q.K^T is (queries x keys x 128 dims),
    # P.V is (queries x 128 dims x keys). Hence macs_pv = macs_qk.
    macs_qk = hpg * nblk * wl.br * wl.bc * D
    macs_pv = macs_qk
    t_array, x, y = split_two_arrays(
        macs_qk, macs_pv,
        mac_rate(a, 16, D), mac_rate(a, 32, D),                        # rates at K=128 (scores)
        mac_rate(a, 16, min(wl.bc, a.kmax)), mac_rate(a, 32, min(wl.bc, a.kmax)))  # at K=256 (outputs)

    # Vector-CPU traffic, item by item. All of it is bookkeeping forced by the
    # hardware (int8 arrays, no in-array accumulation) rather than by attention.
    v = 0.0
    v += 2 * S * D * (wl.kv_bytes + a.sa_in_bytes)            # requantise K and build V^T -- ONCE per group
    v += hpg * S * D * (wl.act_bytes + a.sa_in_bytes)         # requantise Q -- once per head
    v += hpg * nblk * (wl.br * wl.bc * (a.sa_out_bytes + a.sa_in_bytes)   # read scores, write probabilities
                       + wl.br * D * a.sa_out_bytes           # read the P.V partial tile
                       + 2 * wl.br * D * wl.acc_bytes)        # fp32 accumulator read-modify-write
    v += hpg * S * D * (wl.acc_bytes + wl.act_bytes)          # divide by l, write O
    t_vcpu = v / a.bw_sram_vcpu

    # DRAM traffic is tiny: K and V once per group, Q in and O out per head.
    # 5 MB against 2.4 G MACs = 460 MAC/byte, twenty times over the ridge.
    dram = 2 * S * D * wl.kv_bytes + hpg * S * D * wl.act_bytes * 2
    t_dram = dram / a.bw_dma

    per_group = max(t_array, t_vcpu, t_dram)
    n_groups = wl.batch * m.n_kv_heads                        # 16 x 8 = 128 pairs
    cycles = n_groups * per_group + fill_latency(a)
    ph = Phase("attention_prefill", cycles, n_groups * t_array, n_groups * t_dram,
               n_groups * t_vcpu, n_groups * (macs_qk + macs_pv), n_groups * dram, n_groups * v)
    ph.split = (x, y)          # type: ignore[attr-defined]   # stashed so main() can print it
    ph.n_blocks = nblk         # type: ignore[attr-defined]
    return ph


def attention_decode_layer(a: Arch, wl: Workload, m: LlamaModel, ctx: int) -> Phase:
    """Time the decode attention kernel for one layer (Part B of the pseudocode).

    Note `t_array = 0.0` in the returned Phase. That is not an omission -- it
    is the design decision of assumption A10. Decode attention does not use
    the systolic arrays AT ALL, because with a single query row they would
    waste >= 75% of their input port on padding, and K/V would need
    requantising and transposing every step since nothing is ever reused.

    Instead the vector CPU reads each K/V byte once through the widest port on
    the chip. It needs 4L cycles against the DMA's 8L, so it keeps up at half
    the rate and never becomes the limit.

    Everything here is a byte count. There is no compute in this function
    worth modelling, which is precisely what "memory-bound" means."""
    hpg = m.heads_per_group
    kv = 2 * ctx * m.d_head * wl.kv_bytes                    # K and V for the whole context
    v = kv + 2 * hpg * m.d_head * wl.act_bytes               # the vector CPU also touches q and o
    t_dma, t_vcpu = kv / a.bw_dma, v / a.bw_sram_vcpu        # 8L cycles vs 4L cycles
    n_groups = wl.batch * m.n_kv_heads
    macs = n_groups * hpg * 2 * ctx * m.d_head
    cycles = n_groups * max(t_dma, t_vcpu) + fill_latency(a)
    return Phase("attention_decode", cycles, 0.0, n_groups * t_dma, n_groups * t_vcpu,
                 macs, n_groups * kv, n_groups * v)


def elementwise_bytes_per_token(m: LlamaModel, wl: Workload) -> float:
    """Vector-CPU traffic per token per layer for the small ops between matmuls.

    Only the ops that CANNOT be fused into a neighbouring pass appear here:

      RMSNorm  -- absent, because it is fused into the int8 requantisation
                  that has to happen before each matmul anyway. Free.
      RoPE     -- read and write q, k
      residual -- two adds: read a, read b, write, twice
      SwiGLU   -- SiLU(gate)*up, fused into the down-projection's
                  requantisation: read gate, read up, write int8

    Fusing is the point. These ops do almost no arithmetic; their entire cost
    is moving bytes, so every pass avoided is the whole saving."""
    rope = 2 * wl.act_bytes * (m.d_q + m.d_kv)
    residual = 2 * 3 * wl.act_bytes * m.d_model
    swiglu = (2 * wl.act_bytes + 1) * m.d_ff
    return rope + residual + swiglu


def layer_phases(a: Arch, wl: Workload, m: LlamaModel, M: int, decode: bool, ctx: int):
    """Assemble one complete transformer layer as a list of phases.

    Order: QKV projection -> attention -> O projection -> gate+up -> down -> element-wise.

    Two things to notice:

    * `decode` picks which attention kernel to use AND sets act_in_sram on the
      matmuls. One boolean switches the entire model between its two regimes.

    * `extra_dram_bytes=kv_write` on the QKV projection: the new K and V have
      to be appended to the cache in DRAM. Small, but it is real traffic and
      in decode every byte of DRAM traffic is time.

    Phases are returned as a list and later SUMMED, not max'd -- within a phase
    the engines overlap (idea 3), but the phases themselves run one after
    another (assumption A13)."""
    phases = []
    kv_write = M * 2 * m.d_kv * wl.kv_bytes
    for name, K, N in m.layer_gemms():
        if name == "qkv_proj":
            phases.append(gemm(a, wl, name, M, K, N, act_in_sram=decode, extra_dram_bytes=kv_write))
            # attention slots in right after QKV, before the O projection
            if decode:
                phases.append(attention_decode_layer(a, wl, m, ctx))
            else:
                phases.append(attention_prefill_layer(a, wl, m))
        else:
            phases.append(gemm(a, wl, name, M, K, N, act_in_sram=decode))
    ew = M * elementwise_bytes_per_token(m, wl)
    phases.append(Phase("elementwise", ew / a.bw_sram_vcpu + a.lat_ctrl_vcpu, t_vcpu=ew / a.bw_sram_vcpu,
                        vcpu_bytes=ew))
    return phases


def prefill(a: Arch, wl: Workload, m: LlamaModel):
    """The whole prefill = TTFT.

    M = 16 sequences x 2048 tokens = 32768 tokens through every layer at once.
    That enormous M is what makes prefill compute-bound.

    The LM head runs on only the 16 LAST tokens -- you need one next-token
    prediction per sequence, not 32768. Cheap, and easy to get wrong."""
    M = wl.batch * wl.prompt
    per_layer = layer_phases(a, wl, m, M, decode=False, ctx=wl.prompt)
    lm_head = gemm(a, wl, "lm_head(last tokens)", wl.batch, m.d_model, m.vocab, act_in_sram=True)
    total = m.n_layers * sum(p.cycles for p in per_layer) + lm_head.cycles
    return total, per_layer, lm_head


def decode_step(a: Arch, wl: Workload, m: LlamaModel, ctx: int):
    """One decode step = one new token for every sequence.

    M = 16, not 32768. Every weight in the model is read to produce 16 tokens.
    That is the memory-bound regime, and it is why this function's answer is
    essentially just (bytes / 64 GB/s)."""
    per_layer = layer_phases(a, wl, m, wl.batch, decode=True, ctx=ctx)
    lm_head = gemm(a, wl, "lm_head", wl.batch, m.d_model, m.vocab, act_in_sram=True)
    total = m.n_layers * sum(p.cycles for p in per_layer) + lm_head.cycles
    return total, per_layer, lm_head


def decode_avg_step(a: Arch, wl: Workload, m: LlamaModel) -> float:
    """Average step over all 256 generated tokens.

    The context GROWS as generation proceeds (2049 -> 2304), so the KV cache
    read grows too and each step is slightly slower than the last. Rather than
    approximate, evaluate all 256 and average: 185 ms first, 189 average,
    194 last."""
    steps = [decode_step(a, wl, m, wl.prompt + t)[0] for t in range(1, wl.gen + 1)]
    return sum(steps) / len(steps)


# ------------------------------------------------------------------------------------
# 5. Elasticity -- the machinery that answers Problem 2
#
#    "What is the bottleneck?" is vague when several resources are partly busy.
#    Elasticity makes it operational: if I spend money widening THIS wire, does
#    the runtime actually move?
#
#        e(p) = (dT/T) / (dp/p)
#
#        e = -1     time is inversely proportional to p -> p IS the bottleneck
#        e =  0     p is irrelevant, there is slack
#        e = -0.5   p matters but shares the blame
#
#    The method is brute force and all the better for it: perturb the
#    parameter +-10%, re-run the ENTIRE model, measure. No analysis, no
#    assumption about what should matter. This is only possible because every
#    phase is a pure function of an Arch -- which is why the file is built
#    that way.
# ------------------------------------------------------------------------------------
PARAMS = ["bw_dram_dma", "bw_dma_sram", "bw_sram_vcpu", "sa16_in", "sa16_out", "sa32_in",
          "sa32_out", "lat_ctrl_dma", "lat_ctrl_vcpu", "lat_ctrl_sa", "lat_dram_rd", "lat_dram_wr"]


# Some parameters are physically linked and scaling one alone is meaningless:
# the two DRAM links are in series, and buying "half a wider array port" is
# not a thing you can order. Scaling them as groups is what produces the
# report's headline -1.01 figures, as opposed to the misleading -0.55 you get
# from widening one DRAM link on its own.
COMBINED = {
    "DRAM path (both links)": ["bw_dram_dma", "bw_dma_sram"],
    "all array ports": ["sa16_in", "sa16_out", "sa32_in", "sa32_out"],
    "array input ports": ["sa16_in", "sa32_in"],
    "array output ports": ["sa16_out", "sa32_out"],
}


def scaled(a: Arch, params, factor: float) -> Arch:
    """Return a COPY of the chip with some parameters multiplied.

    `replace` (a dataclass copy) rather than mutation -- so the baseline Arch
    is never disturbed and the comparisons stay honest."""
    return replace(a, **{p: getattr(a, p) * factor for p in params})


def elasticity(fn, a: Arch, param, eps: float = 0.10) -> float:
    """Measure (dT/T)/(dp/p) by central difference at +-10%.

    `fn` is anything that turns an Arch into a time -- one kernel, one phase,
    or the whole model. So the same three lines answer "what binds this
    kernel?" and "what binds the entire workload?", which is how the report's
    table gets both its inner and outer columns.

    Central difference (both up AND down, divided by 2*eps) rather than a
    one-sided difference: the model has max() in it and is therefore piecewise
    linear, so sampling both sides is more robust near a kink where the
    bottleneck changes hands."""
    params = [param] if isinstance(param, str) else param
    base = fn(a)
    up = fn(scaled(a, params, 1 + eps))
    dn = fn(scaled(a, params, 1 - eps))
    return (up - dn) / (base * 2 * eps)


# ------------------------------------------------------------------------------------
# 6. Four accelerators on one host -- Problem 5
#
#    The binding constraint: accelerators CANNOT talk to each other directly.
#    Every exchange goes up to the host and back down, on the critical path.
# ------------------------------------------------------------------------------------
def tp_shard(m: LlamaModel, tp: int) -> LlamaModel:
    """Megatron-style tensor parallelism, expressed as a smaller model.

    Elegant trick: instead of writing a separate distributed model, just
    DIVIDE THE MODEL DIMENSIONS BY 4 and re-run the existing single-chip
    model. Each accelerator genuinely does hold a quarter of the heads, a
    quarter of the FFN, a quarter of the vocabulary -- so a quarter-sized
    LlamaModel describes its share exactly.

    Note n_kv_heads: 8 // 4 = 2, exactly. The GQA groups divide cleanly by 4,
    so the decode kernel runs unmodified on each accelerator's own groups.
    Had it not divided, the whole plan would need rethinking."""
    return replace(m, n_heads=m.n_heads // tp, n_kv_heads=m.n_kv_heads // tp,
                   d_ff=m.d_ff // tp, vocab=m.vocab // tp)


def tp_decode(a: Arch, wl: Workload, m: LlamaModel, tp: int, host_bw_Bps: float,
              host_lat_s: float = 0.0):
    """A decode step under tensor parallelism, including host-mediated all-reduces.

    Two all-reduces per layer -- after the O projection and after the down
    projection -- so 64 per token for 32 layers. Each moves
    batch x d_model x 2 bytes = 128 KiB UP to the host and the same back DOWN.

    The `2 *` in t_comm is that round trip, and it is SEQUENTIAL: the host
    cannot return a sum before all four partials have arrived (A12). Assuming
    that serialisation makes the answer conservative, which is the right
    direction for a requirement.

    Passing host_bw_Bps = 0 means "communication is free" -- that is how
    section 5.2 computes the theoretical maximum."""
    ms = tp_shard(m, tp)
    t_comp = a.sec(decode_avg_step(a, wl, ms))     # each chip runs a quarter-model
    n_ar = 2 * m.n_layers                          # 64 all-reduces per token
    ar_bytes = wl.batch * m.d_model * wl.act_bytes # 128 KiB each
    t_comm = n_ar * (2 * ar_bytes / host_bw_Bps + 2 * host_lat_s) if host_bw_Bps > 0 else 0.0
    return t_comp, t_comm, n_ar, ar_bytes


def host_bw_for_fraction(a: Arch, wl: Workload, m: LlamaModel, tp: int, frac: float,
                         host_lat_s: float = 0.0) -> float:
    """Answer 5.3 by working BACKWARDS from a target.

    Not "how fast is this link?" but "how slow may it be and still hit 70%?".

        1. compute-only step is t_comp, so 70% of maximum means the step may
           take t_comp / 0.7
        2. the communication budget is whatever is left over
        3. divide by the number of transfers -> time per all-reduce
        4. bytes / that time -> the required bandwidth

    Answer at 70%: ~0.82 GB/s per accelerator. Roughly PCIe 3.0 x4 -- an
    entirely ordinary link, which is the reassuring conclusion. The workload
    is memory-bound, so the activations that need exchanging (128 KiB) are
    trivial next to the weights being streamed (1.9 GB).

    Returns inf if the budget is non-positive -- i.e. the target is
    unreachable at ANY bandwidth, because compute alone already exceeds it."""
    t_comp, _, n_ar, ar_bytes = tp_decode(a, wl, m, tp, host_bw_Bps=0.0)
    budget = t_comp / frac - t_comp - n_ar * 2 * host_lat_s
    if budget <= 0:
        return math.inf
    return n_ar * 2 * ar_bytes / budget      # bytes/s per accelerator link


# ------------------------------------------------------------------------------------
# 7. Reporting
#
#    Everything below is printing. The model is finished; main() just calls it
#    with different inputs and formats the answers into the report's tables.
# ------------------------------------------------------------------------------------
def fmt_bytes(b: float) -> str:
    for unit, div in (("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if b >= div:
            return f"{b / div:.2f} {unit}"
    return f"{b:.0f} B"


def print_phase_table(a: Arch, phases, title: str):
    """Print one layer as a table: time, then each engine, then who won.

    Showing all three engine columns beside the winner is the whole reason the
    report's bottleneck claims are checkable. A reader can see that prefill's
    DRAM column reads 21, 10, 15, 67, 79 ms against array times 20-40x larger,
    and conclude for themselves that memory bandwidth is nowhere near binding."""
    print(f"\n  {title}")
    print(f"  {'phase':<22}{'time':>10}{'array':>10}{'dram':>10}{'vcpu':>10}  bound")
    tot = 0.0
    for p in phases:
        tot += p.cycles
        print(f"  {p.name:<22}{a.sec(p.cycles)*1e3:>9.2f}ms{a.sec(p.t_array)*1e3:>9.2f}ms"
              f"{a.sec(p.t_dram)*1e3:>9.2f}ms{a.sec(p.t_vcpu)*1e3:>9.2f}ms  {p.bound}")
    print(f"  {'total':<22}{a.sec(tot)*1e3:>9.2f}ms")


def main():
    """Print the whole report, section by section.

        A. derived architectural constants  -> report section 0
        B. workload characterisation        -> report section 0.2
        C. flash-attention kernel           -> Problem 1
        D. elasticity                       -> Problem 2
        E. TTFT and interactivity           -> Problem 3
        F. architectural improvements       -> Problem 4
        G. four accelerators                -> Problem 5
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="dump results to this JSON file")
    ap.add_argument("--plot", action="store_true", help="write roofline.png")
    args = ap.parse_args()

    a, wl, m = Arch(), Workload(), LlamaModel()
    R: dict = {}

    # ---------------- A. derived architectural constants ----------------
    # Not read from a table -- COMPUTED from the bandwidths in Arch. The
    # 32x16/K=128 row showing 512 MAC/cycle (the same as the small array) is
    # the result that drives Problem 2.
    print("=" * 78)
    print("A. DERIVED ARCHITECTURAL CONSTANTS")
    print("=" * 78)
    for rows in (16, 32):
        for K in (128, 256):
            print(f"  {rows}x16 array, K={K:>3}: {tile_cycles(a, rows, K):>6.0f} cycles/tile, "
                  f"{mac_rate(a, rows, K):>6.0f} MAC/cycle "
                  f"(in {(rows+16)*K/(a.sa16_in if rows==16 else a.sa32_in):.0f} cyc, "
                  f"out {rows*16*2/8:.0f} cyc)")
    peak = peak_mac_rate(a)
    print(f"  Peak (both arrays, K=256): {peak:.0f} MAC/cycle = {peak*a.f_hz/1e12:.3f} TMAC/s "
          f"= {2*peak*a.f_hz/1e12:.3f} TOPS")
    # The arrays could ingest weights faster than DRAM can supply them -- which
    # is exactly why decode starves.
    print(f"  Weight ingest capacity of arrays at K=256: {32*256/128 + 16*256/128:.0f} B/cycle "
          f"vs DRAM->SRAM {a.bw_dma:.0f} B/cycle")
    # THE ridge point: 1536 / 64 = 24 MAC per DRAM byte. Everything hinges here.
    print(f"  Ridge point vs DRAM: {peak/a.bw_dma:.1f} MAC per DRAM byte")
    R["peak_mac_per_cycle"] = peak

    # ---------------- B. workload characterisation ----------------
    # The two byte counts that ARE the decode step: 117 ms of weights and
    # 71 ms of KV cache. Their sum is the 189 ms answer to Problem 3.
    print("\n" + "=" * 78)
    print("B. WORKLOAD CHARACTERISATION")
    print("=" * 78)
    print(f"  Linear params (32 layers): {m.linear_params/1e9:.3f} B; LM head {m.lm_head_params/1e9:.3f} B; "
          f"embedding {m.embed_params/1e9:.3f} B; total {m.total_params/1e9:.3f} B")
    w_stream = (m.linear_params + m.lm_head_params) * wl.w_bytes
    print(f"  Weight bytes streamed per decode step: {fmt_bytes(w_stream)} "
          f"-> {w_stream/a.bw_dma/1e6:.1f} ms at 64 GB/s")
    kv_tok = m.kv_bytes_per_token(wl.kv_bytes)
    ctx_avg = wl.prompt + (wl.gen + 1) / 2         # context grows 2049..2304; use the mean
    kv_step = wl.batch * ctx_avg * kv_tok
    print(f"  KV cache: {kv_tok/1024:.0f} KiB/token; batch total at end "
          f"{fmt_bytes(wl.batch*(wl.prompt+wl.gen)*kv_tok)}; read per decode step (avg ctx "
          f"{ctx_avg:.0f}): {fmt_bytes(kv_step)} -> {kv_step/a.bw_dma/1e6:.1f} ms")
    M_pre = wl.batch * wl.prompt
    macs_lin = m.linear_params * M_pre + m.lm_head_params * wl.batch
    # The closed-form TTFT check: total MACs / peak rate = 149 s, against the
    # phase model's 157 s. The 5% gap is real kernel inefficiency, and the two
    # numbers agreeing is the model validating itself.
    print(f"  Prefill linear MACs: {macs_lin:.3e} -> {macs_lin/peak/a.f_hz:.1f} s at peak")
    # 16 MAC/byte against a ridge of 24 -- decode's regime in one line.
    print(f"  Decode arithmetic intensity (GEMM): {wl.batch} MAC / weight byte  (< ridge {peak/a.bw_dma:.0f})")
    R["weights_bytes"] = w_stream
    R["kv_bytes_per_token"] = kv_tok

    # ---------------- C. flash attention kernel ----------------
    # Problem 1's numbers. The optimal split (25% / 100%) is SEARCHED by
    # split_two_arrays, not assumed -- the report cites this as confirmation.
    print("\n" + "=" * 78)
    print("C. FLASH-ATTENTION KERNEL (one layer)")
    print("=" * 78)
    ap_ = attention_prefill_layer(a, wl, m)
    nb = ap_.n_blocks  # type: ignore[attr-defined]
    x, y = ap_.split   # type: ignore[attr-defined]
    print(f"  Prefill: causal blocks per head {nb}/{(wl.prompt//wl.br)*(wl.prompt//wl.bc)} "
          f"({nb/((wl.prompt//wl.br)*(wl.prompt//wl.bc))*100:.1f} %)")
    print(f"  Optimal static split: {x*100:.0f} % of QK^T and {y*100:.0f} % of PV on the 32x16 array")
    print(f"  Effective attention MAC rate: {ap_.macs/ap_.t_array:.0f} MAC/cycle (peak {peak:.0f})")
    print(f"  Per layer: total {a.sec(ap_.cycles)*1e3:.1f} ms | array {a.sec(ap_.t_array)*1e3:.1f} ms | "
          f"vcpu {a.sec(ap_.t_vcpu)*1e3:.1f} ms | dram {a.sec(ap_.t_dram)*1e3:.1f} ms  -> {ap_.bound}-bound")
    print(f"  Utilisation: arrays 100 %, vector CPU {ap_.t_vcpu/ap_.t_array*100:.0f} %, DRAM {ap_.t_dram/ap_.t_array*100:.0f} %")
    ad = attention_decode_layer(a, wl, m, int(ctx_avg))
    print(f"  Decode (ctx {ctx_avg:.0f}): per layer {a.sec(ad.cycles)*1e3:.2f} ms | dram {a.sec(ad.t_dram)*1e3:.2f} ms | "
          f"vcpu {a.sec(ad.t_vcpu)*1e3:.2f} ms -> {ad.bound}-bound; MACs {ad.macs:.2e}")
    R["attn_prefill_layer_ms"] = a.sec(ap_.cycles) * 1e3
    R["attn_decode_layer_ms"] = a.sec(ad.cycles) * 1e3
    R["attn_split"] = (x, y)

    # The 6144-cycle block schedule from the pseudocode, verified here.
    # If the two arrays did NOT come out equal, the kernel's static 3/1 split
    # of the sub-tiles would be wrong and one array would idle.
    print("\n  Steady-state schedule of one (128 q x 256 k) block:")
    s16 = 2 * 16 * tile_cycles(a, 16, m.d_head)      # 32-row S chunk on 16x16: 2 row tiles x 16 key tiles
    s32 = 16 * tile_cycles(a, 32, m.d_head)          # same chunk on 32x16: 16 ops, all 32 rows at once
    pv32 = (m.d_head // 16) * tile_cycles(a, 32, wl.bc)   # PV chunk: 128 channels / 16 per op
    print(f"    S sub-tile (32q x 256k) on 16x16: {s16:.0f} cycles; on 32x16: {s32:.0f} cycles; "
          f"PV sub-tile on 32x16: {pv32:.0f} cycles")
    # THE LOAD-BALANCE CHECK: 3 x 2048 = 6144 on one array, 2048 + 4 x 1024 = 6144
    # on the other. Equal, so neither ever waits for the other.
    print(f"    16x16: 3 S sub-tiles = {3*s16:.0f} cycles | 32x16: 1 S + 4 PV = {s32+4*pv32:.0f} cycles")
    v_blk = 4 * (32 * 256 * 3 + 32 * 128 * 2 + 2 * 32 * 128 * 4)
    print(f"    vector CPU per block: {v_blk} B = {v_blk/a.bw_sram_vcpu:.0f} cycles ({v_blk/a.bw_sram_vcpu/(3*s16)*100:.0f} % busy)")

    # ---------------- D. elasticity ----------------
    # Problem 2, answered by brute force: perturb every parameter, re-run the
    # whole model, print what moved. Four target functions so the same
    # machinery reports both "what binds this kernel" and "what binds the
    # complete workload".
    print("\n" + "=" * 78)
    print("D. ELASTICITY  (dT/T)/(dp/p), central difference +-10 %")
    print("=" * 78)
    targets = {
        "attn prefill": lambda ar: attention_prefill_layer(ar, wl, m).cycles,
        "attn decode": lambda ar: attention_decode_layer(ar, wl, m, int(ctx_avg)).cycles,
        "full prefill": lambda ar: prefill(ar, wl, m)[0],
        "decode step": lambda ar: decode_step(ar, wl, m, int(ctx_avg))[0],
    }
    print(f"  {'parameter':<16}" + "".join(f"{k:>15}" for k in targets))
    R["elasticity"] = {}
    for p in PARAMS:
        row = {k: elasticity(fn, a, p) for k, fn in targets.items()}
        R["elasticity"][p] = row
        print(f"  {p:<16}" + "".join(f"{row[k]:>15.3f}" for k in targets))
    # The combined rows are the ones that matter: scaling linked parameters
    # together turns the misleading -0.55 into the honest -1.01.
    print("  -- combined (parameters scaled together) --")
    for name, ps in COMBINED.items():
        row = {k: elasticity(fn, a, ps) for k, fn in targets.items()}
        R["elasticity"][name] = row
        print(f"  {name:<24}" + "".join(f"{row[k]:>13.3f}" for k in targets))

    # Quantify assumption A7 by simply turning it off. This is where the
    # report's "862 MAC/cycle, 476 ms" figures come from -- and why the zero
    # elasticity of the control latencies is an achievement, not a given.
    a_noq = replace(a, sa_cmd_queue=False)
    print(f"\n  Sensitivity: without a systolic command queue (100 cycles per 128-cycle tile):")
    print(f"    peak {peak_mac_rate(a_noq):.0f} MAC/cycle; attn prefill layer "
          f"{a.sec(attention_prefill_layer(a_noq, wl, m).cycles)*1e3:.0f} ms vs {a.sec(ap_.cycles)*1e3:.0f} ms; "
          f"decode step {a.sec(decode_step(a_noq, wl, m, int(ctx_avg))[0])*1e3:.0f} ms vs "
          f"{a.sec(decode_step(a, wl, m, int(ctx_avg))[0])*1e3:.0f} ms")

    # ---------------- E. TTFT and interactivity ----------------
    # Problem 3. Note the two phase tables: prefill says "arrays" on every row,
    # decode says "dram" on every row. Same code, opposite verdict, purely
    # because M went from 32768 to 16.
    print("\n" + "=" * 78)
    print("E. TTFT AND INTERACTIVITY (single accelerator)")
    print("=" * 78)
    t_pre, pre_layers, pre_lm = prefill(a, wl, m)
    print_phase_table(a, pre_layers, "Prefill, one layer (M = 32768 tokens)")
    print(f"  LM head on last tokens: {a.sec(pre_lm.cycles)*1e3:.1f} ms")
    print(f"  TTFT = {a.sec(t_pre):.1f} s  ({a.sec(t_pre)/60:.2f} min)")
    t_dec_first, dec_layers, dec_lm = decode_step(a, wl, m, wl.prompt + 1)
    print_phase_table(a, dec_layers, f"Decode, one layer (ctx = {wl.prompt+1})")
    print(f"  LM head: {a.sec(dec_lm.cycles)*1e3:.1f} ms")
    t_dec_avg = decode_avg_step(a, wl, m)
    t_dec_last = decode_step(a, wl, m, wl.prompt + wl.gen)[0]
    print(f"  Decode step: first {a.sec(t_dec_first)*1e3:.1f} ms, average {a.sec(t_dec_avg)*1e3:.1f} ms, "
          f"last {a.sec(t_dec_last)*1e3:.1f} ms")
    print(f"  Interactivity: {1/a.sec(t_dec_avg):.2f} tok/s per sequence, "
          f"{wl.batch/a.sec(t_dec_avg):.1f} tok/s aggregate")
    # 157 s of prefill against 48 s of generation: reading the prompt takes
    # longer than writing the whole answer.
    print(f"  End-to-end for the batch: {a.sec(t_pre + wl.gen*t_dec_avg):.1f} s")
    R["ttft_s"] = a.sec(t_pre)
    R["decode_step_avg_ms"] = a.sec(t_dec_avg) * 1e3
    R["tok_per_s_per_seq"] = 1 / a.sec(t_dec_avg)
    R["tok_per_s_aggregate"] = wl.batch / a.sec(t_dec_avg)

    # What if we refused to quantise (assumption A3)? Exact 16-bit multiplies
    # via a hi/lo byte split cost 2x on activation-x-weight matmuls and 4x on
    # attention (activation x activation). Emulated by scaling the array times.
    # Result: TTFT doubles, decode is untouched because it never touches the
    # array limit. Consistent with everything else: array assumptions move
    # prefill only.
    print("\n  Sensitivity to the activation-precision decision (A3):")
    wl2 = replace(wl)
    a2 = replace(a)
    t_pre_exact = 0.0
    for _ in range(m.n_layers):
        for ph in layer_phases(a2, wl2, m, M_pre, decode=False, ctx=wl.prompt):
            f = 4.0 if ph.name.startswith("attention") else (2.0 if ph.t_array > 0 else 1.0)
            t_pre_exact += max(ph.t_array * f, ph.t_dram, ph.t_vcpu * (1.5 if ph.t_array > 0 else 1.0)) + fill_latency(a2)
    print(f"    exact hi/lo-byte decomposition instead of INT8 requantisation: TTFT ~ {a.sec(t_pre_exact):.0f} s "
          f"(x{t_pre_exact/t_pre:.2f}); decode step unchanged (DRAM-bound)")
    print("    PE-bound array model (1 MAC/PE/cycle, K cycles per tile) would halve the array rates: "
          f"TTFT ~ {a.sec(t_pre)*2:.0f} s")
    R["ttft_exact16_s"] = a.sec(t_pre_exact)

    # ---------------- F. architectural improvements ----------------
    # Problem 4. Each "improvement" is just a modified Arch or Workload fed
    # back through the same model. That is the payoff of building it as pure
    # functions: proposing a hardware change costs one line.
    #
    # Watch rows 1) and 1'): x4 and x8 give the IDENTICAL step time. The model
    # keeps going after DRAM stops binding and reports what binds next -- the
    # array input ports. You could not guess that; you have to compute it.
    print("\n" + "=" * 78)
    print("F. ARCHITECTURAL IMPROVEMENTS FOR INTERACTIVITY (decode step, avg ctx)")
    print("=" * 78)
    variants = {
        "baseline": (a, wl),
        # Improvement 1: widen the memory path. Both links together -- widening
        # one alone is half-wasted (they are in series).
        "1) DRAM+DMA bandwidth x4 (256 GB/s)": (replace(a, bw_dram_dma=4 * a.bw_dram_dma, bw_dma_sram=4 * a.bw_dma_sram), wl),
        "1') DRAM+DMA bandwidth x8": (replace(a, bw_dram_dma=8 * a.bw_dram_dma, bw_dma_sram=8 * a.bw_dma_sram), wl),
        # Improvement 2: move fewer bytes. int4 weights + int8 KV, expanded by
        # the DMA engine on the way into SRAM -- so the arrays and vector CPU
        # see the same int8/int16 as before and the KERNELS NEED NO CHANGES.
        # In the model that is literally two numbers.
        "2) INT4 weights + INT8 KV via DMA dequant": (a, replace(wl, w_bytes=0.5, kv_bytes=1.0)),
        "1)+2)": (replace(a, bw_dram_dma=4 * a.bw_dram_dma, bw_dma_sram=4 * a.bw_dma_sram), replace(wl, w_bytes=0.5, kv_bytes=1.0)),
        # And the next rung: only once the arrays are also widened does the
        # workload finally become array-bound -- and TTFT improve at all.
        "1)+2)+ array/vector ports x2": (replace(a, bw_dram_dma=4 * a.bw_dram_dma, bw_dma_sram=4 * a.bw_dma_sram,
                                                 sa16_in=2 * a.sa16_in, sa32_in=2 * a.sa32_in, sa16_out=2 * a.sa16_out,
                                                 sa32_out=2 * a.sa32_out, bw_sram_vcpu=2 * a.bw_sram_vcpu),
                                         replace(wl, w_bytes=0.5, kv_bytes=1.0)),
    }
    R["improvements"] = {}
    print(f"  {'variant':<44}{'step':>10}{'tok/s/seq':>11}{'aggregate':>11}  dominant phase bound")
    for name, (av, wv) in variants.items():
        t = decode_avg_step(av, wv, m)
        _, phs, _ = decode_step(av, wv, m, int(ctx_avg))
        # Report which engine dominates by TIME, not by phase count -- a
        # bottleneck that shows up in one long phase beats three short ones.
        bounds = {}
        for ph in phs:
            bounds[ph.bound] = bounds.get(ph.bound, 0) + ph.cycles
        dom = max(bounds, key=bounds.get)
        R["improvements"][name] = a.sec(t) * 1e3
        print(f"  {name:<44}{a.sec(t)*1e3:>8.1f}ms{1/a.sec(t):>11.2f}{wl.batch/a.sec(t):>11.1f}  {dom}")
    print("  Ladder: after DRAM is widened, weight streaming becomes bound by the arrays' input ports "
          "(96 B/cycle) and KV streaming by the vector port (128 B/cycle).")

    # ---------------- G. four accelerators ----------------
    # Problem 5. The three sharding schemes are compared on the metric the
    # question actually asks about -- interactivity, not throughput. That
    # distinction is what eliminates pipeline parallelism.
    print("\n" + "=" * 78)
    print("G. FOUR ACCELERATORS ON ONE HOST")
    print("=" * 78)
    # host_bw_Bps=0 -> communication is free -> this is the theoretical maximum.
    t_comp, _, n_ar, ar_bytes = tp_decode(a, wl, m, 4, host_bw_Bps=0.0)
    ms4 = tp_shard(m, 4)
    _, tp_layers, tp_lm = decode_step(a, wl, ms4, int(ctx_avg))
    print_phase_table(a, tp_layers, "TP=4 decode, one layer per accelerator (ctx avg)")
    # 47.6 ms = exactly a quarter of 189. Linear scaling, which is what you
    # expect for a purely memory-bound job split perfectly four ways.
    print(f"  TP=4 compute-only step: {t_comp*1e3:.1f} ms -> theoretical max interactivity "
          f"{1/t_comp:.2f} tok/s per sequence, {wl.batch/t_comp:.0f} tok/s aggregate")
    print(f"  All-reduces per step: {n_ar}, {ar_bytes/1024:.0f} KiB up + {ar_bytes/1024:.0f} KiB down per accelerator each")
    # Data parallel: 4 sequences each, but EVERY chip still streams all 7.5 GB
    # of weights. Only the KV share (38% of the step) is split. Hence 135 ms.
    t_dp = a.sec(decode_avg_step(a, replace(wl, batch=4), m))
    print(f"  DP=4 (4 seqs/accelerator, full weights each): step {t_dp*1e3:.1f} ms -> {1/t_dp:.2f} tok/s/seq")
    # Pipeline parallel: the classic trap. Throughput x4, interactivity
    # UNCHANGED, because a single token still has to cross all 32 layers in
    # sequence. Answering the wrong metric would have made this look good.
    print(f"  PP=4: step = single-accelerator step {a.sec(t_dec_avg)*1e3:.1f} ms (+3 hops) -> "
          f"{1/a.sec(t_dec_avg):.2f} tok/s/seq, throughput x4 only")
    # 5.3's answer, plus the 90% figure that shows how steep the cost curve
    # gets near the ceiling: 4x the bandwidth for 20 more points.
    for frac in (0.7, 0.9):
        bw = host_bw_for_fraction(a, wl, m, 4, frac)
        print(f"  Host link bandwidth for {frac*100:.0f} % of max: {bw/1e9:.3f} GB/s per accelerator "
              f"({bw*8/1e9:.1f} Gbit/s); aggregate {4*bw/1e9:.2f} GB/s")
    bw70 = host_bw_for_fraction(a, wl, m, 4, 0.7)
    # Sensitivities: a headline requirement nobody has stress-tested is not a
    # requirement. Latency barely matters; doubling the partial-sum width
    # doubles the answer exactly.
    for lat in (5e-6, 20e-6):
        print(f"    with {lat*1e6:.0f} us per-transfer latency: {host_bw_for_fraction(a, wl, m, 4, 0.7, lat)/1e9:.3f} GB/s")
    print(f"    with 32-bit partial sums: {2*bw70/1e9:.3f} GB/s per accelerator")
    # PREFILL WANTS THE OPPOSITE PARTITION. Under TP the activations in flight
    # are 32768 tokens instead of 16, so each all-reduce carries 256 MiB rather
    # than 128 KiB -- and the communication costs MORE than the compute it
    # saves. So: data-parallel prefill (zero comms), tensor-parallel decode,
    # with a one-off KV re-shard between them.
    tp_pre = a.sec(prefill(a, wl, ms4)[0])
    dp_pre = a.sec(prefill(a, replace(wl, batch=4), m)[0])
    pre_ar_bytes = wl.batch * wl.prompt * m.d_model * wl.act_bytes     # 256 MiB, not 128 KiB
    tp_pre_comm = n_ar * 2 * pre_ar_bytes / bw70
    kv_reshuffle = 0.75 * wl.batch * wl.prompt * kv_tok / 4 / bw70     # 3/4 of each chip's cache, one way
    print(f"  Prefill: TP=4 {tp_pre:.1f} s compute + {tp_pre_comm:.1f} s all-reduce at {bw70/1e9:.2f} GB/s;"
          f" DP=4 {dp_pre:.1f} s + KV re-shard {kv_reshuffle:.1f} s")
    R["tp4"] = {"t_comp_ms": t_comp * 1e3, "max_tok_s_seq": 1 / t_comp, "host_bw_70_GBps": bw70 / 1e9,
                "dp_prefill_s": dp_pre, "tp_prefill_s": tp_pre, "kv_reshuffle_s": kv_reshuffle}

    if args.json:
        with open(args.json, "w") as f:
            json.dump(R, f, indent=2, default=str)
        print(f"\n  wrote {args.json}")
    if args.plot:
        plot_roofline(a, wl, m, peak)


def plot_roofline(a: Arch, wl: Workload, m: LlamaModel, peak: float):
    """Draw Figure 2: the roofline.

    The plot is one line -- min(intensity x 64, 1536) -- on log-log axes:

        the SLOPE  is the DRAM limit  (you cannot compute faster than data arrives)
        the FLAT   is the array limit (you cannot compute faster than the arrays go)
        the CORNER is the ridge, at 24 MAC/byte

    Then the workload's four phases are plotted as dots. The whole argument of
    the report is visual here: decode's two dots sit far left on the slope,
    prefill's two sit far right on the flat, and they are separated by orders
    of magnitude. Same model, same hardware, two completely different machines.

    Note the y-coordinates use macs/cycles -- the ACHIEVED rate from the phase
    model, not the theoretical ceiling. So a dot sitting below the roofline is
    the model honestly reporting kernel inefficiency."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    ai = np.logspace(-1, 5, 400)
    dram = np.minimum(ai * a.bw_dma, peak)         # the roofline itself, in one line
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.loglog(ai, dram, lw=2, label="attainable MAC/cycle = min(intensity x 64 B/cycle, 1536)")
    ax.axvline(peak / a.bw_dma, ls=":", lw=1, color="gray")
    ax.text(peak / a.bw_dma * 1.1, 12, f"ridge {peak/a.bw_dma:.0f} MAC/B", fontsize=8, color="gray")
    ax.text(0.12, 500, "LEFT of ridge: DRAM->DMA->SRAM\n(64 B/cycle) is the limit\n"
                       "-> time = bytes / 64 GB/s  (decode)", fontsize=7.5, color="tab:blue")
    ax.text(40, 400, "RIGHT of ridge: systolic-array ports are the limit\n"
                     "-> time = MACs / 1536 per cycle  (prefill)", fontsize=7.5, color="tab:blue")
    ax.axhspan(peak * 0.98, peak * 1.02, color="tab:blue", alpha=0.08)
    ctx = wl.prompt + (wl.gen + 1) // 2
    ap_ = attention_prefill_layer(a, wl, m)
    ad_ = attention_decode_layer(a, wl, m, ctx)
    # The same GEMM function twice, differing ONLY in M -- and landing on
    # opposite sides of the ridge. This pair of dots is the report's thesis.
    g_dec = gemm(a, wl, "g", wl.batch, m.d_model, 2 * m.d_ff, act_in_sram=True)
    g_pre = gemm(a, wl, "g", wl.batch * wl.prompt, m.d_model, 2 * m.d_ff)
    pts = {
        "decode GEMM (M=16)": (g_dec.macs / g_dec.dram_bytes, g_dec.macs / g_dec.cycles),
        "decode attention (vector CPU)": (ad_.macs / ad_.dram_bytes, ad_.macs / ad_.cycles),
        "prefill GEMM (M=32768)": (g_pre.macs / g_pre.dram_bytes, g_pre.macs / g_pre.cycles),
        "prefill flash-attention": (ap_.macs / ap_.dram_bytes, ap_.macs / ap_.cycles),
    }
    offsets = [(6, -12), (6, -12), (6, 8), (-30, -16)]
    for (name, (x, y)), off in zip(pts.items(), offsets):
        ax.plot(x, y, "o")
        ax.annotate(name, (x, y), textcoords="offset points", xytext=off, fontsize=8)
    ax.set_xlabel("arithmetic intensity  [MAC per DRAM byte]")
    ax.set_ylabel("attainable MAC / cycle")
    ax.set_title("Figure 2 - roofline of the Figure 1 accelerator (1536 MAC/cycle, 64 B/cycle DRAM)", fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig("roofline.png", dpi=150)
    print("  wrote roofline.png")


if __name__ == "__main__":
    main()
