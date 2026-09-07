"""
flash_attention_pseudocode.py  --  Problem 1 of section 2.2
=============================================================

Pseudocode (Python-flavoured, NOT meant to execute) of an optimized flash-attention
kernel for the fixed-architecture accelerator.  This is the program that runs on the
CONTROL CPU: it never touches data itself, it only (a) programs DMA descriptors,
(b) enqueues tile commands to the two systolic arrays, (c) launches pre-compiled
vector-CPU programs, and (d) synchronises the three through events.

LLM inference has two phases with opposite characteristics, so the kernel has TWO
entry points that are designed differently:

                         flash_attention_prefill          flash_attention_decode
  queries per sequence   2048 (the whole prompt)          1 (the new token)
  keys per sequence      2048, causal                     context L = 2049 .. 2304
  MACs per (b,g), layer  4 x 2 x 0.5625 x 2048^2 x 128 = 2.4 G    4 x 2 x L x 128 = 2.2 M
  KV bytes per (b,g)     1 MiB, read once, reused 8192x  L x 512 B = 1.1 MB, read once, used once
  MAC per DRAM byte      ~ 460   -> compute-bound         ~ 2     -> memory-bound
  engines                both arrays 100 %, vector 38 %, DMA 5 %     DMA 100 %, vector 50 %, arrays idle
  design objective       keep both arrays saturated       keep the DRAM->DMA->SRAM pipe saturated
  time per layer         226 ms                           2.2 ms (average context)
  metric it drives       TTFT                             interactivity

Numbers quoted in comments are produced by perf_model.py (section C).

--------------------------------------------------------------------------------------
0.  Runtime primitives assumed to exist on the control CPU  (assumptions A5-A8)
--------------------------------------------------------------------------------------
  dma.copy(dst, src, nbytes, rows=1, src_stride=0, dst_stride=0, after=None) -> Event
        2-D strided copy between DRAM and SRAM (either direction). Issue latency
        200 cycles; descriptors are *queued* so back-to-back copies stream at
        64 B/cycle with the 200-cycle DRAM read latency paid once per descriptor.
  sa16.matmul(A, B, C, K, out_shift, c_stride, after=None) -> Event
  sa32.matmul(A, B, C, K, out_shift, c_stride, after=None) -> Event
        C[rows_A x 16] (int16) = shift_right( A[rows_A x K] . B[16 x K]^T , out_shift )
        A and B are 8-bit rows of K contiguous elements in SRAM (K <= 256).
        rows_A = 16 (sa16) or 32 (sa32).  Commands are queued (FIFO); the 100-cycle
        control->array latency is therefore paid once per burst, and the array is
        pipelined: inputs of tile n+1 stream while outputs of tile n drain.
        Steady-state cost = max(in_bytes/in_bw, out_bytes/out_bw):
              sa16, K=128 :  64 cycles      sa16, K=256 : 128 cycles
              sa32, K=128 : 128 cycles      sa32, K=256 : 128 cycles
  vcpu.run(program, **args, after=None) -> Event
        Launch a resident vector program (300-cycle issue latency, hidden by queueing).
        The vector CPU is memory bound: cost = bytes moved / 128 B/cycle.
  wait(*events) / join(*events) / wait_all()
  Event objects passed as `after=` let the control CPU enqueue far ahead of execution
  (it never spins on a tile).

======================================================================================
PART A  --  PREFILL KERNEL   (compute-bound: the systolic arrays are the resource)
======================================================================================

--------------------------------------------------------------------------------------
A1.  Numerical scheme  (assumption A3)
--------------------------------------------------------------------------------------
  Activations are 16-bit, the arrays take 8-bit inputs and emit 16-bit outputs.
  Every operand fed to an array is dynamically requantised to INT8 by the vector CPU:
      Q  : per query row       q8 = round(q / sq_i),      sq_i = max|q_i| / 127
      K  : per key row         k8 = round(k / sk_j),      sk_j = max|k_j| / 127
      V^T: per (channel c, 256-key block) sv_c^blk        (V is transposed on the fly)
      P  : per query row, probabilities in (0,1] -> uint8 with fixed scale 1/255
  Array outputs are right-shifted so the 256-deep sums fit 16 bits:
      S = Q8.K8^T over d=128 : |sum| <= 127*127*128 = 2.06e6  -> out_shift = 6
      O = P8.V8  over 256 keys: |sum| <= 255*127*256 = 8.29e6 -> out_shift = 8
  The vector CPU undoes the shifts and scales in FP32:
      s_ij = S_raw * 2^6 * sq_i * sk_j / sqrt(128)
      o_ic += O_raw * 2^8 * (1/255) * sv_c^blk        (accumulated in FP32)
  Online softmax (FlashAttention-2 style) keeps m_i (running max) and l_i (running
  sum); the rescale factor alpha_i = exp(m_old - m_new) is applied when the PV partial
  tile of the same block is folded into the accumulator, so it costs no extra traffic.

--------------------------------------------------------------------------------------
A2.  Operand layout  (why both operands are "rows of K contiguous bytes")
--------------------------------------------------------------------------------------
  The arrays compute C = A^T B with A^T and B stored as [rows x K]; i.e. C[i][j] is the
  dot product of row i of A^T with row j of B.  Everything is therefore expressed as
  "rows dot rows":
      S  = Q  . K^T  : A^T = 32 query rows (128 B each),   B = 16 key rows (128 B each)
      O  = P  . V    : A^T = 32 P rows (256 keys each),    B = 16 rows of V^T (256 keys each)
  KV cache in DRAM is token-major for both K and V ([tokens x 128], 16-bit), which
  makes the per-token append in decode a single contiguous 256 B write.  The prefill
  kernel builds the INT8 V^T copy in SRAM once per (sequence, KV group) and amortises
  it over 4 heads x 2048 queries.

--------------------------------------------------------------------------------------
A3.  Tiling  (block sizes are dictated by the array geometry)
--------------------------------------------------------------------------------------
  BC = 256 keys      = Kmax of the arrays (PV contracts over keys)
  BR = 128 queries   = 4 sub-tiles of 32 rows (sa32 geometry); 4 sub-tiles per block
                       is exactly what balances the two arrays (see schedule below)
  D  = 128           = head dim = contraction depth of QK^T  (fixed by the model)
  One (BR x BC) block = 4 S sub-tiles + 4 PV sub-tiles:
      S  sub-tile (32q x 256k, K=128): 32 ops on sa16 (2 row-tiles x 16 key-tiles) = 2048 cyc
                                       or 16 ops on sa32                             = 2048 cyc
      PV sub-tile (32q x 128d, K=256):  8 ops on sa32 (128 d / 16)                  = 1024 cyc
  Static split:  sa16 -> S sub-tiles 0,1,2 (6144 cycles)
                 sa32 -> S sub-tile 3 (2048) + PV sub-tiles 0..3 of the previous block (4096)
                 => both arrays busy 6144 cycles per block, 1365 MAC/cycle (89 % of peak)
                 vector CPU: 256 KiB per block = 2048 cycles (33 % busy)
  Causal: blocks with kj*BC > (qi+1)*BR-1 are skipped (72 of 128 blocks remain);
  partially masked diagonal blocks are computed in full by the arrays and masked by
  the vector CPU before the exponential.

--------------------------------------------------------------------------------------
A4.  SRAM allocation, prefill (16 MiB available; ~4.4 MiB used)
--------------------------------------------------------------------------------------
  region                                         size        lifetime / purpose
  KV_STAGE[2]  K,V of one (b,g), 16-bit           2 x 1 MiB   DMA landing, double-buffered over (b,g)
  K8           [2048 x 128] int8                  256 KiB     current group
  V8T          [128 x 2048] int8                  256 KiB     current group, transposed
  SK, SV       per-key / per-(chan,blk) scales    12 KiB
  Q_STAGE[2]   Q of one head, 16-bit               2 x 512 KiB DMA landing, double-buffered over heads
  Q8[2]        [2048 x 128] int8 + SQ              2 x 264 KiB current / next head
  S_BUF[2]     [128 x 256] int16                   2 x 64 KiB  array output, per block (ping-pong)
  P8_BUF[2]    [128 x 256] uint8                   2 x 32 KiB  vector output -> sa32 input
  OP_BUF[2]    [128 x 128] int16                   2 x 32 KiB  PV partial tiles
  O_ACC        [128 x 128] fp32                    64 KiB      accumulator of the current query block
  M_L          [128] x (m,l,alpha) fp32            2 KiB
  O_OUT[2]     [128 x 128] int16                   2 x 32 KiB  normalised output, DMA to DRAM
  SEM / DESC   semaphores, descriptor ring         8 KiB
  ------------------------------------------------------------------------------------
  total ~ 4.4 MiB.  The remaining ~11.5 MiB is deliberately left free: it lets the
  surrounding GEMMs keep an 8 MiB weight slab resident (perf_model.slab_bytes) and
  gives head-room for prefetching the *next* group's K8/V8T if measurements show the
  requantisation pass is not fully hidden.

--------------------------------------------------------------------------------------
A5.  Dataflow / pipeline (steady state, one key block j of one query block)
--------------------------------------------------------------------------------------
  cycle:        0        2048       4096       6144
  sa16    | S(sub0,j) | S(sub1,j) | S(sub2,j) |
  sa32    | S(sub3,j) | PV(sub0,j-1) PV(sub1,j-1) PV(sub2,j-1) PV(sub3,j-1) |
  vcpu    |     softmax(sub3,j) softmax(sub0,j)  ... acc(sub0..3, j-1) ...   |  (33 % busy)
  dma     |  prefetch next head's Q / next group's K,V ; write back finished O        |  (5 % busy)
  P8 of block j is consumed by sa32 during block j+1 (one-block software pipelining),
  hence the double buffers.  O_ACC <- alpha_j * O_ACC + PV(j) happens when PV(j) lands.


======================================================================================
PART B  --  DECODE KERNEL   (memory-bound: the DRAM->DMA->SRAM pipe is the resource)
======================================================================================

--------------------------------------------------------------------------------------
B1.  Why the decode kernel is a different program  (assumption A10)
--------------------------------------------------------------------------------------
  Per (sequence b, KV group g) and layer: 4 query vectors (the 4 heads of the group,
  1 token each) against L = 2049..2304 keys and values.
      MACs       = 2 x 4 x L x 128        ~ 2.2 M
      DRAM bytes = L x 2 x 128 x 2 B      ~ 1.1 MB   (K and V, 16-bit, read ONCE)
      -> 2 MAC per DRAM byte, far below the 24 MAC/B ridge of the machine.
  Whatever computes it, the kernel cannot finish before the DMA has streamed 1.1 MB at
  64 B/cycle = 8 L cycles (17.4 k cycles per (b,g); 2.2 ms per layer for 128 pairs).

  Option 1 - systolic arrays: A^T must be a 16-row tile but only 4 rows (the 4 heads)
      are useful -> <= 25 % of the input port does work; and K/V would have to be
      requantised to INT8 + V transposed EVERY step (they are used once), costing the
      vector CPU 1.6 MB of traffic per (b,g) on top of the stream.  Array time per
      (b,g): 136 ops x 64 cyc (QK^T) + 68 ops x 128 cyc (PV) = 17.4 k cycles on the
      16x16 array = the same as the DMA time, for extra complexity and precision loss.
  Option 2 - vector CPU (chosen): reads every K/V byte once through the widest port
      of the chip (128 B/cycle -> 4 L cycles, i.e. 50 % of the DMA time), works in
      FP32 with no requantisation, needs no transpose, and leaves the arrays free.

  The design objective therefore flips: the DMA queue must NEVER run dry.  The kernel
  is organised as a stream of fixed-size KV chunks through a ring of SRAM buffers, and
  the control CPU keeps the ring full ahead of the vector CPU.

--------------------------------------------------------------------------------------
B2.  Tiling  (chunking of the KV stream)
--------------------------------------------------------------------------------------
  CH = 256 keys per chunk : K chunk 64 KiB + V chunk 64 KiB = 128 KiB (16-bit)
      DMA time per chunk    = 128 KiB / 64 B  = 2048 cycles   <- the pace-setter
      vector time per chunk = 128 KiB / 128 B = 1024 cycles   (50 % busy)
  Online softmax across chunks for the 4 heads: m[4], l[4], o[4][128] in FP32 (2 KiB),
  updated once per chunk; normalised once per (b,g).  The NEW token's own k and v are
  taken directly from SRAM (they were produced by this layer's QKV GEMM) instead of
  being read back from DRAM, which also removes a read-after-write hazard on the cache.

--------------------------------------------------------------------------------------
B3.  SRAM allocation, decode  (< 1 MiB for attention; the rest serves the GEMMs)
--------------------------------------------------------------------------------------
  region                                    size        purpose
  KV_RING[4]   K||V chunk, 16-bit           4 x 128 KiB  DMA landing ring; 3 chunks in flight
  Q_ALL        [16 seq x 4096] int16        128 KiB      output buffer of the QKV GEMM (already there)
  KV_NEW       k,v of the new token         64 KiB       [16 x 2 x 1024] int16, from the QKV GEMM
  ACC          m,l,o for 4 heads, fp32       2 KiB       current (b,g)
  O_ALL        [16 seq x 4096] int16        128 KiB      input buffer of the O-projection GEMM
  ----------------------------------------------------------------------------------
  total ~ 0.83 MiB (vs 4.4 MiB in prefill).  The remaining ~15 MiB is used by the
  decode GEMMs, which stream weights through a double-buffered 2 x 4 MiB slab region
  while the [16 x K] int8 activation tile stays resident.

--------------------------------------------------------------------------------------
B4.  Dataflow (steady state)
--------------------------------------------------------------------------------------
  for each (b, g):                                     128 pairs per layer
      DMA     : chunk 0 .. n_ch-1 of K and V of (b,g) -> ring        (2048 cyc each)
      vector  : per chunk  s = q4 . K_ch^T * scale ; m,l update ; o = alpha*o + p . V_ch
                                                                    (1024 cyc each)
      vector  : fold new token ; o / l -> int16 -> O_ALL[b][heads of g]   (tiny)
  The control CPU enqueues the DMA descriptors of (b,g)+1 while the vector CPU is
  still on (b,g): the ring is always 3 chunks ahead, so the DMA never waits, and the
  first weight slab of the following O-projection GEMM is enqueued right behind the
  last KV chunk so that the DRAM pipe stays busy across the phase boundary.

  cycle:   0        2048       4096       6144       8192  ...
  DMA    | ch0(b,g) | ch1      | ch2      | ch3      | ch4 ...   100 % busy
  vector |          | ch0      | ch1      | ch2      | ch3 ...    50 % busy
  arrays |                  idle (used by the GEMMs before and after)              |
"""

# ======================================================================================
#  Constants
# ======================================================================================
S, D, H, G = 2048, 128, 32, 8            # seq len, head dim, query heads, KV heads
HPG = H // G                             # 4 query heads per KV group (GQA)
BR, BC = 128, 256                        # query block, key block
NQB, NKB = S // BR, S // BC              # 16 query blocks, 8 key blocks
SHIFT_S, SHIFT_O = 6, 8                  # output right-shifts (section 1)
SCALE = 1.0 / (D ** 0.5)                 # softmax temperature 1/sqrt(d)

# SRAM map (byte offsets, see section 4). Helper regions are allocated statically at
# kernel-load time; only the *contents* change per (b, g, h).
sram = SramMap(  # noqa: F821  (pseudocode)
    KV_STAGE=[alloc(1 << 20), alloc(1 << 20)],
    K8=alloc(S * D), V8T=alloc(D * S), SK=alloc(S * 4), SV=alloc(NKB * D * 4),
    Q_STAGE=[alloc(S * D * 2), alloc(S * D * 2)],
    Q8=[alloc(S * D), alloc(S * D)], SQ=[alloc(S * 4), alloc(S * 4)],
    S_BUF=[alloc(BR * BC * 2), alloc(BR * BC * 2)],
    P8_BUF=[alloc(BR * BC), alloc(BR * BC)],
    OP_BUF=[alloc(BR * D * 2), alloc(BR * D * 2)],
    O_ACC=alloc(BR * D * 4), M_L=alloc(BR * 3 * 4),
    O_OUT=[alloc(BR * D * 2), alloc(BR * D * 2)],
)


# ======================================================================================
#  PART A  --  Vector-CPU programs of the prefill kernel (resident). Bytes/128 = cycles.
# ======================================================================================
def VP_REQUANT_KV(stage, K8, V8T, SK, SV):
    """K (16-bit, [S x D]) -> K8 rows + per-row scale.  V -> V8T (transposed) + per
    (channel, 256-key block) scale.  Traffic: read 1 MiB, write 512 KiB -> 12 288 cycles.
    Run once per (b, g); amortised over 4 heads x 2048 queries."""


def VP_REQUANT_Q(stage, Q8, SQ):
    """Per-row INT8 requantisation of one head's Q. read 512 KiB + write 256 KiB -> 6 144 cyc."""


def VP_SOFTMAX_SUBTILE(S_sub, P8_sub, SQ, SK, M_L, qi, kj, sub):
    """Online softmax of one 32 x 256 int16 score sub-tile.
       s   = S_raw * 2^SHIFT_S * sq_i * sk_j * SCALE          (FP32, free compute)
       if diagonal block: s[i][j] = -inf where key > query   (causal mask)
       m_new = max(m_old, rowmax(s));  alpha = exp(m_old - m_new)
       p     = exp(s - m_new);         l = alpha * l + rowsum(p)
       P8    = round(p * 255) as uint8; store alpha in M_L for the accumulate step
       Traffic: read 16 KiB + write 8 KiB = 192 cycles per sub-tile."""


def VP_ACCUMULATE_SUBTILE(OP_sub, O_ACC, M_L, SV, kj, sub):
    """O_ACC[32 x 128] = alpha * O_ACC + OP_raw * 2^SHIFT_O / 255 * sv[kj]    (FP32)
       Traffic: read 8 KiB partial + RMW 2 x 16 KiB accumulator = 320 cycles per sub-tile."""


def VP_FINALIZE(O_ACC, M_L, O_OUT):
    """O = O_ACC / l  -> int16 (activation format); read 64 KiB + write 32 KiB = 768 cyc."""


# ======================================================================================
#  PART A  --  Control-CPU kernel: PREFILL
# ======================================================================================
def flash_attention_prefill(Q_dram, K_dram, V_dram, O_dram, B=16):
    """Q_dram[b][h]  : [S x D] int16 (after RoPE)         O_dram[b][h] : [S x D] int16
       K_dram[b][g]  : [S x D] int16 (after RoPE)         V_dram[b][g] : [S x D] int16
       Loops: (b, g) -> h in group -> query block qi -> key block kj -> sub-tile."""

    # ---- prologue: prefetch K,V of the first group and Q of its first head -----------
    kv_ev = load_kv(b=0, g=0, buf=0)
    q_ev = load_q(b=0, h=0, buf=0)

    for n, (b, g) in enumerate(product(range(B), range(G))):
        kv_buf = n % 2
        # 1. prefetch the NEXT group's K,V into the other staging buffer (DMA is 5 % busy)
        if (b, g) != (B - 1, G - 1):
            nb, ng = divmod(n + 1, G)
            next_kv_ev = load_kv(nb, ng, buf=1 - kv_buf)

        # 2. requantise K -> K8 (+SK), V -> V8T (+SV) for this group  (12 288 cycles)
        rq_kv = vcpu.run(VP_REQUANT_KV, stage=sram.KV_STAGE[kv_buf], K8=sram.K8,
                         V8T=sram.V8T, SK=sram.SK, SV=sram.SV, after=kv_ev)

        for hh in range(HPG):
            h = g * HPG + hh
            q_buf = hh % 2
            # 3. prefetch NEXT head's Q while this head computes
            if hh + 1 < HPG:
                next_q_ev = load_q(b, h + 1, buf=1 - q_buf)
            elif (b, g) != (B - 1, G - 1):
                next_q_ev = load_q(nb, ng * HPG, buf=1 - q_buf)
            rq_q = vcpu.run(VP_REQUANT_Q, stage=sram.Q_STAGE[q_buf],
                            Q8=sram.Q8[q_buf], SQ=sram.SQ[q_buf], after=q_ev)

            for qi in range(NQB):
                n_kb = qi // 2 + 1                     # causal: keys <= (qi+1)*BR-1
                vcpu.run("VP_ZERO", sram.O_ACC, sram.M_L)   # m=-inf, l=0, O=0
                prev_pv = []                            # PV events of block kj-1
                for kj in range(n_kb):
                    sb, pb, ob = (sram.S_BUF[kj % 2], sram.P8_BUF[kj % 2], sram.OP_BUF[kj % 2])
                    # ---- stage 1: S sub-tiles ------------------------------------------
                    # sa16: sub-tiles 0,1,2 ; sa32: sub-tile 3   (2048 cycles each)
                    s_ev = [None] * 4
                    for sub in range(3):
                        s_ev[sub] = issue_S_subtile_sa16(qi, kj, sub, sb, q_buf, after=[rq_q, rq_kv])
                    s_ev[3] = issue_S_subtile_sa32(qi, kj, 3, sb, q_buf, after=[rq_q, rq_kv])
                    # ---- stage 3 (previous block): PV sub-tiles on sa32 ----------------
                    # P8 of block kj-1 was produced by the vector CPU during block kj-1.
                    pv_ev = []
                    if kj > 0:
                        pb_prev = sram.P8_BUF[(kj - 1) % 2]
                        ob_prev = sram.OP_BUF[(kj - 1) % 2]
                        for sub in range(4):
                            pv_ev.append(issue_PV_subtile_sa32(kj - 1, sub, pb_prev, ob_prev,
                                                               after=[soft_ev_prev[sub], s_ev[3]]))
                    # ---- stage 2: online softmax of block kj (vector CPU) --------------
                    soft_ev = [None] * 4
                    for sub in (3, 0, 1, 2):            # in order of S availability
                        soft_ev[sub] = vcpu.run(VP_SOFTMAX_SUBTILE, S_sub=sb + sub * 32 * BC * 2,
                                                P8_sub=pb + sub * 32 * BC, SQ=sram.SQ[q_buf],
                                                SK=sram.SK, M_L=sram.M_L, qi=qi, kj=kj, sub=sub,
                                                after=s_ev[sub])
                    # ---- fold PV(kj-1) into the FP32 accumulator ----------------------
                    for sub in range(4):
                        if kj > 0:
                            vcpu.run(VP_ACCUMULATE_SUBTILE, OP_sub=ob_prev + sub * 32 * D * 2,
                                     O_ACC=sram.O_ACC, M_L=sram.M_L, SV=sram.SV, kj=kj - 1, sub=sub,
                                     after=pv_ev[sub])
                    soft_ev_prev = soft_ev
                # ---- drain: PV of the last key block, then normalise ------------------
                pb_last, ob_last = sram.P8_BUF[(n_kb - 1) % 2], sram.OP_BUF[(n_kb - 1) % 2]
                last_acc = []
                for sub in range(4):
                    pv = issue_PV_subtile_sa32(n_kb - 1, sub, pb_last, ob_last, after=[soft_ev_prev[sub]])
                    last_acc.append(vcpu.run(VP_ACCUMULATE_SUBTILE, OP_sub=ob_last + sub * 32 * D * 2,
                                             O_ACC=sram.O_ACC, M_L=sram.M_L, SV=sram.SV,
                                             kj=n_kb - 1, sub=sub, after=pv))
                fin = vcpu.run(VP_FINALIZE, O_ACC=sram.O_ACC, M_L=sram.M_L,
                               O_OUT=sram.O_OUT[qi % 2], after=last_acc)
                dma.copy(dst=O_dram[b][h] + qi * BR * D * 2, src=sram.O_OUT[qi % 2],
                         nbytes=BR * D * 2, after=fin)
            q_ev = next_q_ev
        kv_ev = next_kv_ev
    wait_all()


# --------------------------------------------------------------------------------------
#  Helpers used above
# --------------------------------------------------------------------------------------
def load_kv(b, g, buf):
    """K and V of one (b, g): 2 x 512 KiB, contiguous token-major rows of 256 B."""
    e1 = dma.copy(dst=sram.KV_STAGE[buf], src=K_dram[b][g], nbytes=S * D * 2)
    e2 = dma.copy(dst=sram.KV_STAGE[buf] + S * D * 2, src=V_dram[b][g], nbytes=S * D * 2)
    return join(e1, e2)


def load_q(b, h, buf):
    return dma.copy(dst=sram.Q_STAGE[buf], src=Q_dram[b][h], nbytes=S * D * 2)


def issue_S_subtile_sa16(qi, kj, sub, s_buf, q_buf, after):
    """32 queries x 256 keys of scores on the 16x16 array: 2 row tiles x 16 key tiles,
    each op = 16 q-rows (128 B) . 16 k-rows (128 B)^T -> 16x16 int16, 64 cycles."""
    q0 = qi * BR + sub * 32
    evs = []
    for r in range(2):
        for kt in range(BC // 16):
            evs.append(sa16.matmul(
                A=sram.Q8[q_buf] + (q0 + r * 16) * D,
                B=sram.K8 + (kj * BC + kt * 16) * D,
                C=s_buf + (sub * 32 + r * 16) * BC * 2 + kt * 16 * 2,
                K=D, out_shift=SHIFT_S, c_stride=BC * 2, after=after))
    return join(*evs)


def issue_S_subtile_sa32(qi, kj, sub, s_buf, q_buf, after):
    """Same sub-tile on the 32x16 array: 16 ops of 32 q-rows . 16 k-rows, 128 cycles each
    (output-port bound at K=128, see report, problem 2)."""
    q0 = qi * BR + sub * 32
    evs = []
    for kt in range(BC // 16):
        evs.append(sa32.matmul(
            A=sram.Q8[q_buf] + q0 * D,
            B=sram.K8 + (kj * BC + kt * 16) * D,
            C=s_buf + sub * 32 * BC * 2 + kt * 16 * 2,
            K=D, out_shift=SHIFT_S, c_stride=BC * 2, after=after))
    return join(*evs)


def issue_PV_subtile_sa32(kj, sub, p8_buf, op_buf, after):
    """O_partial[32 q x 128 d] = P8[32 x 256 keys] . V8T[128 x 256 keys]^T :
    8 ops (16 output channels each) of K=256, 128 cycles each -> 1024 cycles."""
    evs = []
    for ct in range(D // 16):
        evs.append(sa32.matmul(
            A=p8_buf + sub * 32 * BC,
            B=sram.V8T + (ct * 16) * S + kj * BC,           # rows = channels, cols = keys
            C=op_buf + sub * 32 * D * 2 + ct * 16 * 2,
            K=BC, out_shift=SHIFT_O, c_stride=D * 2, after=after))
    return join(*evs)


# ======================================================================================
#  PART B  --  Control-CPU kernel: DECODE  (one new token per sequence)
# ======================================================================================
CH = 256                                  # keys per streamed chunk
NRING = 4                                 # chunk buffers in the DMA landing ring
sram_dec = SramMap(  # noqa: F821  (pseudocode) -- see header B3
    KV_RING=[alloc(2 * CH * D * 2) for _ in range(NRING)],   # 4 x 128 KiB
    ACC=alloc(HPG * (D + 2) * 4),                             # m, l, o[128] per head, fp32
)


def VP_DEC_CHUNK(q4, Kc, Vc, n_keys, ACC, first):
    """One 256-key chunk for the 4 heads of a GQA group, FP32, no requantisation.
         s[h][j] = q4[h] . Kc[j] * SCALE               (4 x n_keys)
         m_new   = max(m[h], max_j s[h][j]); alpha = exp(m[h] - m_new)
         p[h][j] = exp(s[h][j] - m_new);   l[h] = alpha*l[h] + sum_j p
         o[h]    = alpha*o[h] + sum_j p[h][j] * Vc[j]
       Traffic: read Kc + Vc = 128 KiB (q4 and ACC live in registers) -> 1024 cycles."""


def VP_DEC_FINALIZE(q4, k_new, v_new, ACC, o_out):
    """Fold the new token (its k,v come from SRAM, not from the cache), then
       o_out[h] = int16( o[h] / l[h] )  for the 4 heads.  Traffic ~ 3 KiB."""


def flash_attention_decode(Q_ALL, KV_NEW, K_dram, V_dram, ctx, O_ALL, B=16):
    """Q_ALL[b][h]  : [D] int16 query of the new token (output of this layer's QKV GEMM)
       KV_NEW[b][g] : k,v of the new token, int16 (same GEMM); also appended to the
                      DRAM cache by the QKV phase for use in the NEXT steps.
       ctx[b]       : tokens already in the cache for sequence b (2048 .. 2303)
       O_ALL[b][h]  : [D] int16 output, consumed in SRAM by the O-projection GEMM.

    Time per (b,g) = n_ch x 2048 cycles (DMA-bound); per layer 128 x 8.5 x 2048
    = 2.2 M cycles = 2.2 ms at the average context.  See report, problem 1 (B)."""
    ring = RingScheduler(NRING)               # hands out (buf, free_event) in order
    plan = [(b, g, c) for b in range(B) for g in range(G) for c in range(ceil_div(ctx[b], CH))]

    # ---- DMA side: enqueue the whole stream, throttled only by ring-buffer reuse -------
    land = {}
    for (b, g, c) in plan:
        buf, free_ev = ring.next()            # free_ev = vector CPU finished with this buffer
        n_keys = min(CH, ctx[b] - c * CH)
        e1 = dma.copy(dst=sram_dec.KV_RING[buf], src=K_dram[b][g] + c * CH * D * 2,
                      nbytes=n_keys * D * 2, after=free_ev)
        e2 = dma.copy(dst=sram_dec.KV_RING[buf] + CH * D * 2, src=V_dram[b][g] + c * CH * D * 2,
                      nbytes=n_keys * D * 2, after=free_ev)
        land[(b, g, c)] = (buf, n_keys, join(e1, e2))

    # ---- vector side: consume chunks in the same order ---------------------------------
    for b in range(B):
        for g in range(G):
            q4 = Q_ALL[b][g * HPG:(g + 1) * HPG]
            n_ch = ceil_div(ctx[b], CH)
            done = None
            for c in range(n_ch):
                buf, n_keys, landed = land[(b, g, c)]
                done = vcpu.run(VP_DEC_CHUNK, q4=q4, Kc=sram_dec.KV_RING[buf],
                                Vc=sram_dec.KV_RING[buf] + CH * D * 2, n_keys=n_keys,
                                ACC=sram_dec.ACC, first=(c == 0), after=landed)
                ring.release(buf, done)       # buffer may be refilled once this finishes
            vcpu.run(VP_DEC_FINALIZE, q4=q4, k_new=KV_NEW[b][g].k, v_new=KV_NEW[b][g].v,
                     ACC=sram_dec.ACC, o_out=O_ALL[b][g * HPG:(g + 1) * HPG], after=done)
    # The O-projection GEMM of this layer is enqueued right after; its first weight-slab
    # DMA is queued behind the last KV chunk so the DRAM pipe never idles.
    wait_all()
