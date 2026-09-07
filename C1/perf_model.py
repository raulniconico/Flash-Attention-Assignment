#!/usr/bin/env python3
"""
perf_model.py -- Analytical performance model of the assignment's fixed-architecture
AI inference accelerator running Llama 3.1 8B (W8 / A16, batch 16, 2048-token prompt,
256 generated tokens).

Usage
-----
    python3 perf_model.py                 # print every number used in report.md
    python3 perf_model.py --json out.json # also dump raw results
    python3 perf_model.py --plot          # also write roofline.png (matplotlib)

Modelling philosophy (see report.md, section 1 and the assumption list A1..A12)
------------------------------------------------------------------------------
* Every engine is a *port* with a bandwidth in bytes/cycle taken from the assignment
  table (1 cycle = 1 ns at 1 GHz).  Time on an engine = bytes moved / port bandwidth.
* The systolic arrays are characterised only by their SRAM ports.  A tile op with
  contraction depth K costs  max(input_bytes/in_bw, output_bytes/out_bw)  cycles in
  steady state, because the arrays are pipelined (inputs of tile n+1 overlap the
  read-out of tile n).  At K = 256 both arrays need exactly 128 cycles per tile,
  which is clearly how the assignment's numbers were chosen.
* Engines that run concurrently (DMA, the two arrays, the vector CPU) are combined
  with max(): the kernels are double-buffered so the slowest engine hides the others.
  Fixed latencies (control-CPU issue, DRAM first byte) are paid once per phase as
  pipeline-fill cost, because the control program issues commands into queues ahead
  of time (A7).
* The vector CPU has infinite compute (assignment) => its time is bytes / 128.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict, replace

# ------------------------------------------------------------------------------------
# 1. Architecture constants (assignment table, converted to bytes/cycle)
# ------------------------------------------------------------------------------------
@dataclass
class Arch:
    f_hz: float = 1.0e9
    bw_dram_dma: float = 512 / 8       # 64  B/cycle  (DRAM <-> DMA)
    bw_dma_sram: float = 512 / 8       # 64  B/cycle  (DMA  <-> SRAM)
    bw_sram_vcpu: float = 1024 / 8     # 128 B/cycle  (SRAM <-> vector CPU)
    sa16_in: float = 512 / 8           # 64  B/cycle  (SRAM -> 16x16 array inputs)
    sa16_out: float = 64 / 8           # 8   B/cycle  (16x16 array -> SRAM outputs)
    sa32_in: float = 768 / 8           # 96  B/cycle
    sa32_out: float = 64 / 8           # 8   B/cycle
    lat_ctrl_dma: float = 200          # cycles
    lat_ctrl_vcpu: float = 300
    lat_ctrl_sa: float = 100
    lat_dram_rd: float = 200
    lat_dram_wr: float = 150
    sram_bytes: int = 16 << 20
    dram_bytes: int = 64 << 30
    kmax: int = 256                    # max contraction depth per tile op
    sa_in_bytes: int = 1               # 8-bit array inputs
    sa_out_bytes: int = 2              # 16-bit array outputs
    # ---- modelling switch (assumption A7) ----
    sa_cmd_queue: bool = True          # tile commands are queued; 100-cycle latency paid once per burst

    @property
    def bw_dma(self) -> float:
        """Effective DRAM->SRAM streaming bandwidth (DMA sits between two 64 B/cycle links)."""
        return min(self.bw_dram_dma, self.bw_dma_sram)

    def sec(self, cycles: float) -> float:
        return cycles / self.f_hz


# ------------------------------------------------------------------------------------
# 2. Model and workload
# ------------------------------------------------------------------------------------
@dataclass
class LlamaModel:
    name: str = "Llama 3.1 8B"
    n_layers: int = 32
    d_model: int = 4096
    n_heads: int = 32
    n_kv_heads: int = 8
    d_head: int = 128
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
        return self.n_heads // self.n_kv_heads

    # Per-layer GEMMs as (name, K, N) with activations of shape [tokens x K].
    # Weights are stored transposed ([N x K], K contiguous) so that every operand fed to
    # the systolic array is a row of K contiguous 8-bit elements.
    def layer_gemms(self):
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
        return self.n_layers * 2 * self.d_kv * kv_bytes


@dataclass
class Workload:
    batch: int = 16
    prompt: int = 2048
    gen: int = 256
    w_bytes: float = 1.0       # weight element size in DRAM (INT8)
    act_bytes: int = 2         # activation element size in DRAM / between ops (INT16)
    kv_bytes: float = 2.0      # KV-cache element size in DRAM (INT16, per spec)
    acc_bytes: int = 4         # vector-CPU accumulator element (FP32)
    br: int = 128              # flash-attention query block  (4 sub-tiles of 32 rows)
    bc: int = 256              # flash-attention key block    (= Kmax of the arrays)
    slab_bytes: int = 8 << 20  # weight slab kept resident in SRAM during prefill GEMMs


# ------------------------------------------------------------------------------------
# 3. Systolic-array tile model
# ------------------------------------------------------------------------------------
def tile_cycles(a: Arch, rows_a: int, K: int) -> float:
    """Steady-state cycles per tile op on the (rows_a x 16) array with contraction K."""
    in_bw, out_bw = (a.sa16_in, a.sa16_out) if rows_a == 16 else (a.sa32_in, a.sa32_out)
    t_in = (rows_a + 16) * K * a.sa_in_bytes / in_bw
    t_out = rows_a * 16 * a.sa_out_bytes / out_bw
    t = max(t_in, t_out)
    if not a.sa_cmd_queue:
        t += a.lat_ctrl_sa
    return t


def mac_rate(a: Arch, rows_a: int, K: int) -> float:
    """Useful MACs per cycle of one array at contraction depth K."""
    return rows_a * 16 * K / tile_cycles(a, rows_a, K)


def peak_mac_rate(a: Arch) -> float:
    return mac_rate(a, 16, a.kmax) + mac_rate(a, 32, a.kmax)


# ------------------------------------------------------------------------------------
# 4. Phase model
# ------------------------------------------------------------------------------------
@dataclass
class Phase:
    name: str
    cycles: float
    t_array: float = 0.0
    t_dram: float = 0.0
    t_vcpu: float = 0.0
    macs: float = 0.0
    dram_bytes: float = 0.0
    vcpu_bytes: float = 0.0

    @property
    def bound(self) -> str:
        d = {"array": self.t_array, "dram": self.t_dram, "vcpu": self.t_vcpu}
        return max(d, key=d.get)


def fill_latency(a: Arch) -> float:
    """Pipeline-fill cost paid once per phase (commands are queued, A7)."""
    return a.lat_ctrl_dma + a.lat_dram_rd + a.lat_ctrl_sa + a.lat_ctrl_vcpu


def gemm(a: Arch, wl: Workload, name: str, M: int, K: int, N: int,
         act_in_sram: bool = False, extra_dram_bytes: float = 0.0) -> Phase:
    """Y[M x N] = X[M x K] . W[K x N].  Weights stream from DRAM, INT8 at the array.

    Array:  both arrays work on disjoint output tiles; each tile op streams a 16-row
            activation tile and a 16/32-row weight tile through the input port.
    DRAM:   weights once; in prefill the INT8 activation copy is re-read once per
            weight slab that fits in SRAM (2-D tiling, A9).
    Vector: (i) requantise the 16-bit input to INT8 with a per-row scale,
            (ii) accumulate the 16-bit partial tile of every 256-deep K-chunk into an
            FP32 accumulator (the array has no in-place accumulation, A4),
            (iii) final read of the accumulator + write of the 16-bit output.
    """
    M_eff = math.ceil(M / 16) * 16                 # B operand is always a 16-row tile
    Kc = min(K, a.kmax)
    n_kc = math.ceil(K / Kc)
    macs = M * K * N
    macs_issued = M_eff * K * N
    rate = mac_rate(a, 16, Kc) + mac_rate(a, 32, Kc)
    t_array = macs_issued / rate

    w_bytes = K * N * wl.w_bytes
    if act_in_sram:
        act_traffic = 0.0
    else:
        n_slabs = math.ceil(w_bytes / wl.slab_bytes)
        act_traffic = (M * K * wl.act_bytes            # read 16-bit input once
                       + M * K * a.sa_in_bytes         # write INT8 copy
                       + n_slabs * M * K * a.sa_in_bytes   # re-read INT8 copy per slab
                       + M * N * wl.act_bytes)         # write 16-bit output
    dram_bytes = w_bytes + act_traffic + extra_dram_bytes
    t_dram = dram_bytes / a.bw_dma

    requant = M * K * (wl.act_bytes + a.sa_in_bytes)
    partial = n_kc * (M_eff * N * a.sa_out_bytes + 2 * M * N * wl.acc_bytes)
    final = M * N * (wl.acc_bytes + wl.act_bytes)
    vcpu_bytes = requant + partial + final
    t_vcpu = vcpu_bytes / a.bw_sram_vcpu

    cycles = max(t_array, t_dram, t_vcpu) + fill_latency(a)
    return Phase(name, cycles, t_array, t_dram, t_vcpu, macs, dram_bytes, vcpu_bytes)


def causal_blocks(S: int, br: int, bc: int) -> int:
    """Number of (query block, key block) pairs that are not fully masked."""
    n = 0
    for i in range(S // br):
        q_last = (i + 1) * br - 1
        n += sum(1 for j in range(S // bc) if j * bc <= q_last)
    return n


def split_two_arrays(w_qk: float, w_pv: float,
                     r16_qk: float, r32_qk: float, r16_pv: float, r32_pv: float):
    """Optimal static split of QK^T and PV work between the two arrays.

    x = fraction of QK^T work on the 32x16 array, y = fraction of PV work on it.
    Minimise the makespan max(T16, T32).  For fixed y, T16 is decreasing and T32 is
    increasing in x, so the optimum is at their crossing (clipped to [0,1]).
    Returns (cycles, x, y).
    """
    a_, b_ = w_qk / r16_qk, w_pv / r16_pv      # cost of 100 % of each job on the 16x16
    c_, d_ = w_qk / r32_qk, w_pv / r32_pv      # ... and on the 32x16
    best = (math.inf, 0.0, 0.0)
    steps = 1000
    for k in range(steps + 1):
        y = k / steps
        x = (a_ + (1 - y) * b_ - y * d_) / (a_ + c_)
        x = min(1.0, max(0.0, x))
        t16 = (1 - x) * a_ + (1 - y) * b_
        t32 = x * c_ + y * d_
        t = max(t16, t32)
        if t < best[0]:
            best = (t, x, y)
    return best


def attention_prefill_layer(a: Arch, wl: Workload, m: LlamaModel) -> Phase:
    """Flash-attention forward for one layer, all sequences, all heads (prefill).

    Per (sequence b, KV group g): K/V are loaded once and shared by the 4 query heads
    of the group (GQA).  Blocks of BR queries x BC keys are streamed through a
    3-stage pipeline  [QK^T on arrays] -> [online softmax on vector CPU] -> [PV on
    arrays].  See flash_attention_pseudocode.py for the schedule.
    """
    S, D = wl.prompt, m.d_head
    hpg = m.heads_per_group
    nblk = causal_blocks(S, wl.br, wl.bc)
    macs_qk = hpg * nblk * wl.br * wl.bc * D
    macs_pv = macs_qk
    t_array, x, y = split_two_arrays(
        macs_qk, macs_pv,
        mac_rate(a, 16, D), mac_rate(a, 32, D),
        mac_rate(a, 16, min(wl.bc, a.kmax)), mac_rate(a, 32, min(wl.bc, a.kmax)))

    v = 0.0
    v += 2 * S * D * (wl.kv_bytes + a.sa_in_bytes)            # requantise K and V^T (once per group)
    v += hpg * S * D * (wl.act_bytes + a.sa_in_bytes)         # requantise Q (per head)
    v += hpg * nblk * (wl.br * wl.bc * (a.sa_out_bytes + a.sa_in_bytes)   # read S, write P
                       + wl.br * D * a.sa_out_bytes           # read O partial tile
                       + 2 * wl.br * D * wl.acc_bytes)        # FP32 accumulator read-modify-write
    v += hpg * S * D * (wl.acc_bytes + wl.act_bytes)          # final normalisation, write O
    t_vcpu = v / a.bw_sram_vcpu

    dram = 2 * S * D * wl.kv_bytes + hpg * S * D * wl.act_bytes * 2   # K,V in; Q in; O out
    t_dram = dram / a.bw_dma

    per_group = max(t_array, t_vcpu, t_dram)
    n_groups = wl.batch * m.n_kv_heads
    cycles = n_groups * per_group + fill_latency(a)
    ph = Phase("attention_prefill", cycles, n_groups * t_array, n_groups * t_dram,
               n_groups * t_vcpu, n_groups * (macs_qk + macs_pv), n_groups * dram, n_groups * v)
    ph.split = (x, y)          # type: ignore[attr-defined]
    ph.n_blocks = nblk         # type: ignore[attr-defined]
    return ph


def attention_decode_layer(a: Arch, wl: Workload, m: LlamaModel, ctx: int) -> Phase:
    """Decode attention (1 query per sequence) for one layer.

    Runs on the vector CPU: with a single query row the arrays would waste >= 75 % of
    their input port on padding rows, whereas the vector CPU reads each K/V byte once
    through the widest port on the chip (A10).  K/V stream DRAM -> SRAM by DMA.
    """
    hpg = m.heads_per_group
    kv = 2 * ctx * m.d_head * wl.kv_bytes
    v = kv + 2 * hpg * m.d_head * wl.act_bytes
    t_dma, t_vcpu = kv / a.bw_dma, v / a.bw_sram_vcpu
    n_groups = wl.batch * m.n_kv_heads
    macs = n_groups * hpg * 2 * ctx * m.d_head
    cycles = n_groups * max(t_dma, t_vcpu) + fill_latency(a)
    return Phase("attention_decode", cycles, 0.0, n_groups * t_dma, n_groups * t_vcpu,
                 macs, n_groups * kv, n_groups * v)


def elementwise_bytes_per_token(m: LlamaModel, wl: Workload) -> float:
    """Vector-CPU traffic per token per layer for ops not fused into a GEMM pass.
    RMSNorm is fused with the INT8 requantisation that precedes each GEMM (free).
    RoPE: read+write q,k;  two residual adds: read a, read b, write;  SiLU(gate)*up
    fused with the down-proj requantisation: read gate, read up, write INT8."""
    rope = 2 * wl.act_bytes * (m.d_q + m.d_kv)
    residual = 2 * 3 * wl.act_bytes * m.d_model
    swiglu = (2 * wl.act_bytes + 1) * m.d_ff
    return rope + residual + swiglu


def layer_phases(a: Arch, wl: Workload, m: LlamaModel, M: int, decode: bool, ctx: int):
    phases = []
    kv_write = M * 2 * m.d_kv * wl.kv_bytes
    for name, K, N in m.layer_gemms():
        if name == "qkv_proj":
            phases.append(gemm(a, wl, name, M, K, N, act_in_sram=decode, extra_dram_bytes=kv_write))
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
    M = wl.batch * wl.prompt
    per_layer = layer_phases(a, wl, m, M, decode=False, ctx=wl.prompt)
    lm_head = gemm(a, wl, "lm_head(last tokens)", wl.batch, m.d_model, m.vocab, act_in_sram=True)
    total = m.n_layers * sum(p.cycles for p in per_layer) + lm_head.cycles
    return total, per_layer, lm_head


def decode_step(a: Arch, wl: Workload, m: LlamaModel, ctx: int):
    per_layer = layer_phases(a, wl, m, wl.batch, decode=True, ctx=ctx)
    lm_head = gemm(a, wl, "lm_head", wl.batch, m.d_model, m.vocab, act_in_sram=True)
    total = m.n_layers * sum(p.cycles for p in per_layer) + lm_head.cycles
    return total, per_layer, lm_head


def decode_avg_step(a: Arch, wl: Workload, m: LlamaModel) -> float:
    """Average decode-step cycles over the 256 generated tokens (context grows)."""
    steps = [decode_step(a, wl, m, wl.prompt + t)[0] for t in range(1, wl.gen + 1)]
    return sum(steps) / len(steps)


# ------------------------------------------------------------------------------------
# 5. Elasticity analysis
# ------------------------------------------------------------------------------------
PARAMS = ["bw_dram_dma", "bw_dma_sram", "bw_sram_vcpu", "sa16_in", "sa16_out", "sa32_in",
          "sa32_out", "lat_ctrl_dma", "lat_ctrl_vcpu", "lat_ctrl_sa", "lat_dram_rd", "lat_dram_wr"]


COMBINED = {
    "DRAM path (both links)": ["bw_dram_dma", "bw_dma_sram"],
    "all array ports": ["sa16_in", "sa16_out", "sa32_in", "sa32_out"],
    "array input ports": ["sa16_in", "sa32_in"],
    "array output ports": ["sa16_out", "sa32_out"],
}


def scaled(a: Arch, params, factor: float) -> Arch:
    return replace(a, **{p: getattr(a, p) * factor for p in params})


def elasticity(fn, a: Arch, param, eps: float = 0.10) -> float:
    """(dT/T) / (dp/p) by central difference at +-eps.  `param` may be a list."""
    params = [param] if isinstance(param, str) else param
    base = fn(a)
    up = fn(scaled(a, params, 1 + eps))
    dn = fn(scaled(a, params, 1 - eps))
    return (up - dn) / (base * 2 * eps)


# ------------------------------------------------------------------------------------
# 6. Four accelerators on one host
# ------------------------------------------------------------------------------------
def tp_shard(m: LlamaModel, tp: int) -> LlamaModel:
    """Megatron-style tensor parallelism: heads, KV heads, FFN and vocab are split."""
    return replace(m, n_heads=m.n_heads // tp, n_kv_heads=m.n_kv_heads // tp,
                   d_ff=m.d_ff // tp, vocab=m.vocab // tp)


def tp_decode(a: Arch, wl: Workload, m: LlamaModel, tp: int, host_bw_Bps: float,
              host_lat_s: float = 0.0):
    """Decode step under TP with a host-mediated all-reduce (accelerators only talk to
    the host).  Two all-reduces per layer (after o_proj and after down_proj); each one
    moves batch*d_model*act_bytes up to the host and the same amount back down, on the
    critical path (A12)."""
    ms = tp_shard(m, tp)
    t_comp = a.sec(decode_avg_step(a, wl, ms))
    n_ar = 2 * m.n_layers
    ar_bytes = wl.batch * m.d_model * wl.act_bytes
    t_comm = n_ar * (2 * ar_bytes / host_bw_Bps + 2 * host_lat_s) if host_bw_Bps > 0 else 0.0
    return t_comp, t_comm, n_ar, ar_bytes


def host_bw_for_fraction(a: Arch, wl: Workload, m: LlamaModel, tp: int, frac: float,
                         host_lat_s: float = 0.0) -> float:
    t_comp, _, n_ar, ar_bytes = tp_decode(a, wl, m, tp, host_bw_Bps=0.0)
    budget = t_comp / frac - t_comp - n_ar * 2 * host_lat_s
    if budget <= 0:
        return math.inf
    return n_ar * 2 * ar_bytes / budget      # bytes/s per accelerator link


# ------------------------------------------------------------------------------------
# 7. Reporting
# ------------------------------------------------------------------------------------
def fmt_bytes(b: float) -> str:
    for unit, div in (("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if b >= div:
            return f"{b / div:.2f} {unit}"
    return f"{b:.0f} B"


def print_phase_table(a: Arch, phases, title: str):
    print(f"\n  {title}")
    print(f"  {'phase':<22}{'time':>10}{'array':>10}{'dram':>10}{'vcpu':>10}  bound")
    tot = 0.0
    for p in phases:
        tot += p.cycles
        print(f"  {p.name:<22}{a.sec(p.cycles)*1e3:>9.2f}ms{a.sec(p.t_array)*1e3:>9.2f}ms"
              f"{a.sec(p.t_dram)*1e3:>9.2f}ms{a.sec(p.t_vcpu)*1e3:>9.2f}ms  {p.bound}")
    print(f"  {'total':<22}{a.sec(tot)*1e3:>9.2f}ms")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="dump results to this JSON file")
    ap.add_argument("--plot", action="store_true", help="write roofline.png")
    args = ap.parse_args()

    a, wl, m = Arch(), Workload(), LlamaModel()
    R: dict = {}

    # ---------------- derived architectural constants ----------------
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
    print(f"  Weight ingest capacity of arrays at K=256: {32*256/128 + 16*256/128:.0f} B/cycle "
          f"vs DRAM->SRAM {a.bw_dma:.0f} B/cycle")
    print(f"  Ridge point vs DRAM: {peak/a.bw_dma:.1f} MAC per DRAM byte")
    R["peak_mac_per_cycle"] = peak

    # ---------------- workload characterisation ----------------
    print("\n" + "=" * 78)
    print("B. WORKLOAD CHARACTERISATION")
    print("=" * 78)
    print(f"  Linear params (32 layers): {m.linear_params/1e9:.3f} B; LM head {m.lm_head_params/1e9:.3f} B; "
          f"embedding {m.embed_params/1e9:.3f} B; total {m.total_params/1e9:.3f} B")
    w_stream = (m.linear_params + m.lm_head_params) * wl.w_bytes
    print(f"  Weight bytes streamed per decode step: {fmt_bytes(w_stream)} "
          f"-> {w_stream/a.bw_dma/1e6:.1f} ms at 64 GB/s")
    kv_tok = m.kv_bytes_per_token(wl.kv_bytes)
    ctx_avg = wl.prompt + (wl.gen + 1) / 2
    kv_step = wl.batch * ctx_avg * kv_tok
    print(f"  KV cache: {kv_tok/1024:.0f} KiB/token; batch total at end "
          f"{fmt_bytes(wl.batch*(wl.prompt+wl.gen)*kv_tok)}; read per decode step (avg ctx "
          f"{ctx_avg:.0f}): {fmt_bytes(kv_step)} -> {kv_step/a.bw_dma/1e6:.1f} ms")
    M_pre = wl.batch * wl.prompt
    macs_lin = m.linear_params * M_pre + m.lm_head_params * wl.batch
    print(f"  Prefill linear MACs: {macs_lin:.3e} -> {macs_lin/peak/a.f_hz:.1f} s at peak")
    print(f"  Decode arithmetic intensity (GEMM): {wl.batch} MAC / weight byte  (< ridge {peak/a.bw_dma:.0f})")
    R["weights_bytes"] = w_stream
    R["kv_bytes_per_token"] = kv_tok

    # ---------------- flash attention kernel ----------------
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

    # sub-block timing for the pseudocode schedule
    print("\n  Steady-state schedule of one (128 q x 256 k) block:")
    s16 = 2 * 16 * tile_cycles(a, 16, m.d_head)      # one 32-row sub-tile of S on 16x16: 2 row tiles x 16 key tiles
    s32 = 16 * tile_cycles(a, 32, m.d_head)          # same on 32x16
    pv32 = (m.d_head // 16) * tile_cycles(a, 32, wl.bc)
    print(f"    S sub-tile (32q x 256k) on 16x16: {s16:.0f} cycles; on 32x16: {s32:.0f} cycles; "
          f"PV sub-tile on 32x16: {pv32:.0f} cycles")
    print(f"    16x16: 3 S sub-tiles = {3*s16:.0f} cycles | 32x16: 1 S + 4 PV = {s32+4*pv32:.0f} cycles")
    v_blk = 4 * (32 * 256 * 3 + 32 * 128 * 2 + 2 * 32 * 128 * 4)
    print(f"    vector CPU per block: {v_blk} B = {v_blk/a.bw_sram_vcpu:.0f} cycles ({v_blk/a.bw_sram_vcpu/(3*s16)*100:.0f} % busy)")

    # ---------------- elasticity ----------------
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
    print("  -- combined (parameters scaled together) --")
    for name, ps in COMBINED.items():
        row = {k: elasticity(fn, a, ps) for k, fn in targets.items()}
        R["elasticity"][name] = row
        print(f"  {name:<24}" + "".join(f"{row[k]:>13.3f}" for k in targets))
    # what if tile commands are NOT queued?
    a_noq = replace(a, sa_cmd_queue=False)
    print(f"\n  Sensitivity: without a systolic command queue (100 cycles per 128-cycle tile):")
    print(f"    peak {peak_mac_rate(a_noq):.0f} MAC/cycle; attn prefill layer "
          f"{a.sec(attention_prefill_layer(a_noq, wl, m).cycles)*1e3:.0f} ms vs {a.sec(ap_.cycles)*1e3:.0f} ms; "
          f"decode step {a.sec(decode_step(a_noq, wl, m, int(ctx_avg))[0])*1e3:.0f} ms vs "
          f"{a.sec(decode_step(a, wl, m, int(ctx_avg))[0])*1e3:.0f} ms")

    # ---------------- TTFT and interactivity ----------------
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
    print(f"  End-to-end for the batch: {a.sec(t_pre + wl.gen*t_dec_avg):.1f} s")
    R["ttft_s"] = a.sec(t_pre)
    R["decode_step_avg_ms"] = a.sec(t_dec_avg) * 1e3
    R["tok_per_s_per_seq"] = 1 / a.sec(t_dec_avg)
    R["tok_per_s_aggregate"] = wl.batch / a.sec(t_dec_avg)

    # exact-16-bit variant (A3 alternative): activations split in hi/lo bytes
    print("\n  Sensitivity to the activation-precision decision (A3):")
    wl2 = replace(wl)
    a2 = replace(a)
    # 2x MACs for act x weight GEMMs, 4x for act x act (attention) -> emulate via rates
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

    # ---------------- architectural improvements ----------------
    print("\n" + "=" * 78)
    print("F. ARCHITECTURAL IMPROVEMENTS FOR INTERACTIVITY (decode step, avg ctx)")
    print("=" * 78)
    variants = {
        "baseline": (a, wl),
        "1) DRAM+DMA bandwidth x4 (256 GB/s)": (replace(a, bw_dram_dma=4 * a.bw_dram_dma, bw_dma_sram=4 * a.bw_dma_sram), wl),
        "1') DRAM+DMA bandwidth x8": (replace(a, bw_dram_dma=8 * a.bw_dram_dma, bw_dma_sram=8 * a.bw_dma_sram), wl),
        "2) INT4 weights + INT8 KV via DMA dequant": (a, replace(wl, w_bytes=0.5, kv_bytes=1.0)),
        "1)+2)": (replace(a, bw_dram_dma=4 * a.bw_dram_dma, bw_dma_sram=4 * a.bw_dma_sram), replace(wl, w_bytes=0.5, kv_bytes=1.0)),
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
        bounds = {}
        for ph in phs:
            bounds[ph.bound] = bounds.get(ph.bound, 0) + ph.cycles
        dom = max(bounds, key=bounds.get)
        R["improvements"][name] = a.sec(t) * 1e3
        print(f"  {name:<44}{a.sec(t)*1e3:>8.1f}ms{1/a.sec(t):>11.2f}{wl.batch/a.sec(t):>11.1f}  {dom}")
    print("  Ladder: after DRAM is widened, weight streaming becomes bound by the arrays' input ports "
          "(96 B/cycle) and KV streaming by the vector port (128 B/cycle).")

    # ---------------- four accelerators ----------------
    print("\n" + "=" * 78)
    print("G. FOUR ACCELERATORS ON ONE HOST")
    print("=" * 78)
    t_comp, _, n_ar, ar_bytes = tp_decode(a, wl, m, 4, host_bw_Bps=0.0)
    ms4 = tp_shard(m, 4)
    _, tp_layers, tp_lm = decode_step(a, wl, ms4, int(ctx_avg))
    print_phase_table(a, tp_layers, "TP=4 decode, one layer per accelerator (ctx avg)")
    print(f"  TP=4 compute-only step: {t_comp*1e3:.1f} ms -> theoretical max interactivity "
          f"{1/t_comp:.2f} tok/s per sequence, {wl.batch/t_comp:.0f} tok/s aggregate")
    print(f"  All-reduces per step: {n_ar}, {ar_bytes/1024:.0f} KiB up + {ar_bytes/1024:.0f} KiB down per accelerator each")
    t_dp = a.sec(decode_avg_step(a, replace(wl, batch=4), m))
    print(f"  DP=4 (4 seqs/accelerator, full weights each): step {t_dp*1e3:.1f} ms -> {1/t_dp:.2f} tok/s/seq")
    print(f"  PP=4: step = single-accelerator step {a.sec(t_dec_avg)*1e3:.1f} ms (+3 hops) -> "
          f"{1/a.sec(t_dec_avg):.2f} tok/s/seq, throughput x4 only")
    for frac in (0.7, 0.9):
        bw = host_bw_for_fraction(a, wl, m, 4, frac)
        print(f"  Host link bandwidth for {frac*100:.0f} % of max: {bw/1e9:.3f} GB/s per accelerator "
              f"({bw*8/1e9:.1f} Gbit/s); aggregate {4*bw/1e9:.2f} GB/s")
    bw70 = host_bw_for_fraction(a, wl, m, 4, 0.7)
    for lat in (5e-6, 20e-6):
        print(f"    with {lat*1e6:.0f} us per-transfer latency: {host_bw_for_fraction(a, wl, m, 4, 0.7, lat)/1e9:.3f} GB/s")
    print(f"    with 32-bit partial sums: {2*bw70/1e9:.3f} GB/s per accelerator")
    tp_pre = a.sec(prefill(a, wl, ms4)[0])
    dp_pre = a.sec(prefill(a, replace(wl, batch=4), m)[0])
    pre_ar_bytes = wl.batch * wl.prompt * m.d_model * wl.act_bytes
    tp_pre_comm = n_ar * 2 * pre_ar_bytes / bw70
    kv_reshuffle = 0.75 * wl.batch * wl.prompt * kv_tok / 4 / bw70   # per accelerator, one direction
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
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    ai = np.logspace(-1, 5, 400)
    dram = np.minimum(ai * a.bw_dma, peak)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.loglog(ai, dram, lw=2, label=f"DRAM 64 GB/s roof, array peak {peak:.0f} MAC/cycle")
    ax.axvline(peak / a.bw_dma, ls=":", lw=1, color="gray")
    ax.text(peak / a.bw_dma * 1.1, 12, f"ridge {peak/a.bw_dma:.0f} MAC/B", fontsize=8, color="gray")
    ctx = wl.prompt + (wl.gen + 1) // 2
    ap_ = attention_prefill_layer(a, wl, m)
    ad_ = attention_decode_layer(a, wl, m, ctx)
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
    ax.set_title("Roofline of the accelerator for Llama 3.1 8B phases")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig("roofline.png", dpi=150)
    print("  wrote roofline.png")


if __name__ == "__main__":
    main()
