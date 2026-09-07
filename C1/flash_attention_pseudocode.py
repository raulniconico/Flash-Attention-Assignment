"""
flash_attention_pseudocode.py  --  Task 2.2.1
===============================================

Pseudocode (Python-flavoured, NOT meant to execute) of an optimized flash-attention
kernel for the fixed-architecture accelerator.  This is the program that runs on the
CONTROL CPU: it never touches data itself, it only (a) programs DMA descriptors,
(b) enqueues tile commands to the two systolic arrays, (c) launches pre-compiled
vector-CPU programs, and (d) synchronises the three through events/semaphores.

Two entry points:
    flash_attention_prefill(...)   Sq = Sk = 2048, causal, arrays + vector CPU
    flash_attention_decode(...)    Sq = 1 per sequence, vector CPU only (see A10)

Numbers quoted in comments are produced by perf_model.py (section C).

--------------------------------------------------------------------------------------
0.  Runtime primitives assumed to exist on the control CPU  (assumptions A5-A8)
--------------------------------------------------------------------------------------
  dma.copy(dst, src, nbytes, rows=1, src_stride=0, dst_stride=0) -> Event
        2-D strided copy between DRAM and SRAM (either direction). Issue latency
        200 cycles; descriptors are *queued* so back-to-back copies stream at
        64 B/cycle with the 200-cycle DRAM read latency paid once per descriptor.
  sa16.matmul(A, B, C, K, out_shift, c_stride) -> Event
  sa32.matmul(A, B, C, K, out_shift, c_stride) -> Event
        C[rows_A x 16] (int16) = shift_right( A[rows_A x K] . B[16 x K]^T , out_shift )
        A and B are 8-bit rows of K contiguous elements in SRAM (K <= 256).
        rows_A = 16 (sa16) or 32 (sa32).  Commands are queued (FIFO); the 100-cycle
        control->array latency is therefore paid once per burst, and the array is
        pipelined: inputs of tile n+1 stream while outputs of tile n drain.
        Steady-state cost = max(in_bytes/in_bw, out_bytes/out_bw):
              sa16, K=128 :  64 cycles      sa16, K=256 : 128 cycles
              sa32, K=128 : 128 cycles      sa32, K=256 : 128 cycles
  vcpu.run(program, **args) -> Event
        Launch a resident vector program (300-cycle issue latency, hidden by queueing).
        The vector CPU is memory bound: cost = bytes moved / 128 B/cycle.
  wait(*events) / ev.done()
  Event objects can also be passed as `after=` dependencies to any primitive so the
  control CPU can enqueue far ahead of execution (it never spins on a tile).

--------------------------------------------------------------------------------------
1.  Numerical scheme  (assumption A3)
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
2.  Operand layout  (why both operands are "rows of K contiguous bytes")
--------------------------------------------------------------------------------------
  The arrays compute C = A^T B with A^T and B stored as [rows x K]; i.e. C[i][j] is the
  dot product of row i of A^T with row j of B.  Everything is therefore expressed as
  "rows dot rows":
      S  = Q  . K^T  : A^T = 32 query rows (128 B each),   B = 16 key rows (128 B each)
      O  = P  . V    : A^T = 32 P rows (256 keys each),    B = 16 rows of V^T (256 keys each)
  KV cache in DRAM is token-major for both K and V ([tokens x 128], 16-bit), which
  makes the per-token append a single contiguous 256 B write.  The prefill kernel
  builds the INT8 V^T copy in SRAM once per (sequence, KV group) and amortises it over
  4 heads x 2048 queries.

--------------------------------------------------------------------------------------
3.  Tiling  (block sizes are dictated by the array geometry)
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
4.  SRAM allocation (16 MiB available; ~4.4 MiB used, prefill)
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
5.  Dataflow / pipeline (steady state, one key block j of one query block)
--------------------------------------------------------------------------------------
  cycle:        0        2048       4096       6144
  sa16    | S(sub0,j) | S(sub1,j) | S(sub2,j) |
  sa32    | S(sub3,j) | PV(sub0,j-1) PV(sub1,j-1) PV(sub2,j-1) PV(sub3,j-1) |
  vcpu    |     softmax(sub3,j) softmax(sub0,j)  ... acc(sub0..3, j-1) ...   |  (33 % busy)
  dma     |  prefetch next head's Q / next group's K,V ; write back finished O        |  (5 % busy)
  P8 of block j is consumed by sa32 during block j+1 (one-block software pipelining),
  hence the double buffers.  O_ACC <- alpha_j * O_ACC + PV(j) happens when PV(j) lands.
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
#  Vector-CPU programs (resident; launched by the control CPU). Bytes/128 = cycles.
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


def VP_DECODE_ATTENTION(q, K_stage, V_stage, ctx, o):
    """All 4 heads of one (b, g) for a single query token, entirely on the vector CPU
       in FP32 (no requantisation needed):  s = q.K^T * SCALE, softmax, o = p.V.
       Traffic = read K and V once (ctx * 512 B) + q/o (2 KiB): ~4 cycles per key."""


# ======================================================================================
#  Control-CPU kernel: PREFILL
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
    (output-port bound at K=128, see report section 4)."""
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
#  Control-CPU kernel: DECODE  (one new token per sequence)
# ======================================================================================
def flash_attention_decode(q_sram, K_dram, V_dram, ctx, o_sram, B=16):
    """q_sram[b][h] : [D] int16 for the new token (already in SRAM from the QKV GEMM)
       ctx[b]       : current context length of sequence b (2048 .. 2304)
       o_sram[b][h] : [D] int16 output, consumed in SRAM by the O-projection GEMM.

    Design decision (A10): with Sq = 1 the arrays would stream 16 rows of A^T for 4
    useful ones (GQA group) and reach <= 128 MAC/cycle per array (<= 256 combined), while the vector CPU reads the
    same K/V bytes once through the widest port on the chip (128 B/cycle).  The kernel is
    DRAM-bound anyway (DMA 64 B/cycle < vector 128 B/cycle): 2.2 ms per layer at the
    average context, versus 0.4 ms for the whole QKV projection.  See report section 5.

    SRAM: KV_STAGE[2] (2 x 1.2 MiB, sized for ctx <= 2304) double-buffered over (b, g)."""
    n = 0
    ev = load_kv_prefix(0, 0, ctx[0], buf=0)
    for b in range(B):
        for g in range(G):
            buf = n % 2
            if (b, g) != (B - 1, G - 1):
                nb, ng = divmod(n + 1, G)
                next_ev = load_kv_prefix(nb, ng, ctx[nb], buf=1 - buf)   # DMA runs ahead
            vcpu.run(VP_DECODE_ATTENTION, q=q_sram[b][g * HPG:(g + 1) * HPG],
                     K_stage=sram.KV_STAGE[buf], V_stage=sram.KV_STAGE[buf] + ctx[b] * D * 2,
                     ctx=ctx[b], o=o_sram[b][g * HPG:(g + 1) * HPG], after=ev)
            ev = next_ev
            n += 1
    wait_all()


def load_kv_prefix(b, g, ctx_b, buf):
    e1 = dma.copy(dst=sram.KV_STAGE[buf], src=K_dram[b][g], nbytes=ctx_b * D * 2)
    e2 = dma.copy(dst=sram.KV_STAGE[buf] + ctx_b * D * 2, src=V_dram[b][g], nbytes=ctx_b * D * 2)
    return join(e1, e2)
