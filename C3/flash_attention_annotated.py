"""
flash_attention_annotated.py
=============================================================================
A plain-language rewrite of flash_attention_pseudocode.py.

Same algorithm, same numbers, same structure -- but every block is explained
from first principles instead of in shorthand. Read the original for the
compact version; read this one to understand WHY each line is there.

Nothing here executes. It is a description of a program.


=============================================================================
STEP 0 -- WHAT IS THIS CHIP, AND WHO DOES THE WORK?
=============================================================================

Picture three workers who can all work AT THE SAME TIME:

  DMA           The delivery truck.
                Moves data between big-slow memory (DRAM) and small-fast
                memory (SRAM). 64 bytes per cycle.

  sa16, sa32    The muscle. Two "systolic arrays".
                They do exactly one thing: multiply matrices. Nothing else.
                No exp(), no divide, no compare.

  vector CPU    The handyman.
                Does everything the arrays cannot: exponentials, division,
                converting between number formats, comparisons.
                Reads/writes 128 bytes per cycle -- the widest port on the chip.

And a fourth participant who does NO work at all:

  control CPU   The manager. This file.
                It never touches a single number. It only hands out jobs and
                says "start this one AFTER that one finishes".


How the manager hands out a job
-------------------------------
Every job is submitted and returns immediately with an Event -- a receipt.

    e = sa16.matmul(...)        # "do this multiply"  -> receipt e
    vcpu.run(prog, after=e)     # "do this, but not until e is done"

The manager NEVER waits. It dumps hundreds of jobs into queues with their
dependencies attached and walks away. The hardware figures out the ordering.
This is why the code can look like it is doing things out of order: it is
building a dependency graph, not executing steps.


The one hardware fact that shapes everything
--------------------------------------------
    sa16.matmul(A, B, C, K, ...)   computes  C[16 x 16] = A[16 x K] . B[16 x K]^T
    sa32.matmul(A, B, C, K, ...)   computes  C[32 x 16] = A[32 x K] . B[16 x K]^T

    with K <= 256.

Read that carefully. The output tile size is WIRED IN. It is not a parameter.
An sa32 multiply always costs 128 cycles -- whether you gave it 32 useful rows
or 1 useful row and 31 rows of garbage.

Almost every design decision below follows from this one sentence.

Cost of one multiply in steady state (the arrays are pipelined, so the
100-cycle command latency is paid once per burst, not once per op):

        sa16, K=128 :  64 cycles          sa16, K=256 : 128 cycles
        sa32, K=128 : 128 cycles          sa32, K=256 : 128 cycles

Note row 2: at K=128 the two arrays cost the SAME (sa32 is output-port
limited). At K=256, sa32 does twice the work for the same price. Remember
this -- it is why the two arrays get different jobs later.


=============================================================================
STEP 1 -- THE TWO PHASES ARE TWO DIFFERENT PROBLEMS
=============================================================================

An LLM does two very different things, and they stress opposite parts of
the chip. So this file has two entry points, designed independently.

PREFILL -- reading the user's 2048-word prompt, all at once.
    Per (sequence, KV group), per layer:
        math       : 2.4 billion multiply-accumulates
        data moved : ~1 MiB of keys and values
        ratio      : ~460 MAC per byte from DRAM

    460 is a huge number. The chip's break-even ("ridge") is 24. So the
    delivery truck is nearly idle and the arrays are the bottleneck.
    -> This is COMPUTE-BOUND.
    -> Design goal: never let an array sit idle. Not for one cycle.
    -> Result: 226 ms per layer. This is what makes the user wait for the
       first word to appear (TTFT).

DECODE -- generating one new word.
    Per (sequence, KV group), per layer:
        math       : 2.2 million MACs      (1000x less)
        data moved : ~1.1 MB              (slightly MORE)
        ratio      : ~2 MAC per byte

    2 is far below the ridge of 24. Now the truck is the bottleneck and the
    arrays have nothing useful to do.
    -> This is MEMORY-BOUND.
    -> Design goal: never let the DRAM pipe run dry.
    -> Result: 2.2 ms per layer. This is the typing speed the user sees.

Same mathematical operation. Opposite engineering problem. Hence two kernels.


=============================================================================
STEP 2 -- WHAT ATTENTION ACTUALLY COMPUTES  (the 3 steps)
=============================================================================

For each query (word) we ask: how much should I look at each earlier word?
Then take a weighted average of their values.

    step 1   S = Q . K^T          "score every word against every word"
    step 2   P = softmax(S)       "turn scores into percentages summing to 1"
    step 3   O = P . V            "weighted average of the values"

Map that onto the workers:

    step 1  -> matrix multiply    -> the ARRAYS
    step 2  -> needs exp()        -> the VECTOR CPU
    step 3  -> matrix multiply    -> the ARRAYS

So the pattern is:  array -> handyman -> array.
Keeping all three busy at once is the entire scheduling problem.


=============================================================================
STEP 3 -- WHY TILE, AND WHY THESE TILE SIZES?
=============================================================================

We cannot do 2048 x 2048 in one go. Two separate reasons, often confused:

WHY TILE THE KEYS (BC = 256)?   -> because of MEMORY.
    The full score matrix S is 2048 x 2048 x 2 bytes = 8 MiB. SRAM is 16 MiB
    total and other things need it. So we never build S. We process 256 keys
    at a time and carry a small running summary forward. That trick is what
    the word "flash" in FlashAttention refers to. With tiling, the score
    buffer is 128 x 256 x 2 = 64 KiB instead of 8 MiB.

    Why exactly 256? Because step 3 (P . V) sums over keys, so keys become
    the "K" dimension of the multiply, and the array's limit is K <= 256.
    We pick the largest legal value.

WHY TILE THE QUERIES (BR = 128)?  -> because of the ARRAY SHAPE.
    Recall: an sa32 op processes 32 rows whether you fill them or not.
    Feed it one query at a time and you waste 31/32 of the machine.
    So queries are processed 32 at a time minimum.

    Why 128 (= 4 chunks of 32) and not 64 or 256? Because 4 is exactly the
    number that makes the two arrays finish at the same instant. See STEP 5.

WHY NOT TILE THE 128 FEATURE CHANNELS?
    Because D=128 is the dimension being SUMMED OVER in step 1. Splitting a
    sum means producing partial results and adding them later = extra memory
    traffic. And 128 already fits under the K <= 256 limit in one shot.
    There is nothing to gain.

So: keys tiled to bound memory, queries tiled to fill the array, features
not tiled at all.


=============================================================================
STEP 4 -- WHY EVERYTHING IS CONVERTED TO 8-BIT (and back)
=============================================================================

The arrays only accept 8-bit inputs and only emit 16-bit outputs. But the
model's activations are 16-bit. So somebody must convert, and that somebody
is the vector CPU. This is pure overhead forced by the hardware.

Converting to 8 bits ("quantising") means: find the largest value, divide
everything by (largest / 127), round. Store the divisor so you can undo it.

    Q    : one divisor per query row
    K    : one divisor per key row
    V    : one divisor per (channel, 256-key block)     [V is also transposed]
    P    : probabilities are already in (0, 1], so a fixed divisor 1/255
           works and no search is needed

The 16-bit output can also overflow, so the array right-shifts before writing:

    S = Q8 . K8^T summing 128 terms : max |sum| = 127*127*128 = 2.06e6
        2.06e6 does not fit in int16 (max 32767) -> shift right by 6
    O = P8 . V8  summing 256 terms  : max |sum| = 255*127*256 = 8.29e6
        -> shift right by 8

The vector CPU then undoes shift and divisors in FP32, where it is free
(the vector CPU is memory-bound, so arithmetic costs nothing):

    real_score  = S_raw * 2^6 * q_divisor * k_divisor / sqrt(128)
    real_output = O_raw * 2^8 * (1/255)  * v_divisor       (accumulated FP32)


=============================================================================
STEP 5 -- THE CORE TRICK: RUN STEP 3 ONE TILE BEHIND
=============================================================================

Naive schedule for one tile:

    arrays  | scores |          | output |
    vcpu    |        | softmax  |

The arrays go idle during softmax. In a compute-bound phase, an idle array
is money set on fire.

Fix: do not finish tile 5 before starting tile 6. Deliberately keep step 3
one tile behind, so there is always array work available:

    while doing SCORES  for tile 5
       also doing OUTPUT for tile 4        <-- note: 4, not 5
       and the handyman does SOFTMAX for tile 5

In code this is a single "- 1":

    if kj > 0:
        issue_PV_subtile_sa32(kj - 1, ...)

That minus-one is the heart of the whole kernel.

Consequence: tile 4's data is still being read while tile 5's is being
written. So every buffer exists TWICE and they alternate. That is what all
the [kj % 2] indexing is. Two whiteboards: present one while erasing the other.


THE ARRAY SPLIT
---------------
Each tile needs 4 score-chunks and 4 output-chunks (of 32 queries each).
Recall from STEP 0:
    score chunk (K=128): costs the SAME on both arrays -> 2048 cycles
    output chunk (K=256): sa32 is TWICE as fast        -> 1024 cycles on sa32

So give sa32 the work only it is good at:

    sa16  ->  3 score chunks                        = 3 x 2048 = 6144 cycles
    sa32  ->  1 score chunk + 4 output chunks       = 2048 + 4x1024 = 6144 cycles

Both finish at 6144. Neither ever waits for the other. That is 1365 MAC per
cycle = 89% of the machine's peak. And it is why BR is 128: with any other
number of chunks the split does not balance.

Meanwhile the vector CPU handles 256 KiB per tile = 2048 cycles, so it is
only 33% busy, and the DMA about 5% busy. They have slack; the arrays do not.
Everything is arranged around the arrays.


THE STEADY-STATE PICTURE
------------------------
  cycle:      0          2048        4096        6144
  sa16    | S(chunk0) | S(chunk1) | S(chunk2) |
  sa32    | S(chunk3) | PV(chunk0..3 of the PREVIOUS tile)              |
  vcpu    |   softmax of this tile ... accumulate of previous tile ...  |  33%
  dma     |   prefetch next head's Q / next group's K,V ; write back O  |   5%


=============================================================================
STEP 6 -- CAUSALITY: SKIP WORK YOU DO NOT NEED
=============================================================================

Word #100 cannot look at word #500 -- it has not been said yet. So roughly
the upper-right half of the score matrix is never needed.

Query tile qi covers queries [qi*128 .. qi*128+127].
The highest key it may attend to is qi*128+127.
Number of 256-key tiles needed = (qi*128+127) // 256 + 1 = qi // 2 + 1.

Summed over the 16 query tiles: 1+1+2+2+...+8+8 = 72 tiles instead of 128.
That is the 0.5625 factor in the MAC count at the top of the original file.

Tiles that straddle the diagonal are computed IN FULL by the arrays and then
masked by the vector CPU before the exponential. Masking on the arrays is
impossible (they only multiply), and the vector CPU has spare capacity, so
this costs nothing.


=============================================================================
STEP 7 -- ONLINE SOFTMAX (why tiling keys does not break the math)
=============================================================================

Normal softmax needs ALL the scores before it can start: you must know the
maximum (for numerical stability) and the total (to divide by). But we only
ever hold 256 keys at a time. Contradiction?

No -- keep a running summary per query row and fix up as you go:

    m  = biggest score seen so far
    l  = running total of exp(score - m)
    O  = running weighted sum of values

When a new tile arrives with a bigger maximum, everything accumulated so far
was computed against the old maximum and is now scaled wrong. Correct it:

    m_new = max(m_old, max of this tile)
    alpha = exp(m_old - m_new)          <-- the correction factor
    l     = alpha * l + (sum over this tile)
    O     = alpha * O + (contribution of this tile)

Divide O by l once at the very end. Mathematically identical to plain softmax.

The clever part in this kernel: applying alpha to O looks like an extra pass
over the accumulator -- but the accumulator is being read and written anyway
to add the new tile's contribution. So alpha rides along for free. That is
the "FlashAttention-2 style" note in the original.


=============================================================================
STEP 8 -- SRAM BUDGET FOR PREFILL   (16 MiB available, ~4.4 MiB used)
=============================================================================

  region                        size          why it exists
  ---------------------------------------------------------------------------
  KV_STAGE[2]   K,V 16-bit      2 x 1 MiB     truck unloads here; 2 copies so
                                              the next group loads while this
                                              one computes
  K8            int8            256 KiB       converted K, whole 2048 keys,
                                              stays resident for 4 heads
  V8T           int8            256 KiB       converted AND transposed V
  SK, SV        divisors        12 KiB        needed to undo the conversion
  Q_STAGE[2]    Q 16-bit        2 x 512 KiB   same double-buffer, over heads
  Q8[2] + SQ    int8            2 x 264 KiB   converted Q for current/next head
  S_BUF[2]      int16           2 x 64 KiB    array writes scores here
  P8_BUF[2]     uint8           2 x 32 KiB    vcpu writes probabilities here,
                                              array reads them next tile
  OP_BUF[2]     int16           2 x 32 KiB    array writes partial outputs
  O_ACC         fp32            64 KiB        the running O (must be FP32:
                                              it is summed over many tiles)
  M_L           fp32            2 KiB         the m, l, alpha per query row
  O_OUT[2]      int16           2 x 32 KiB    finished result, truck picks up
  ---------------------------------------------------------------------------
  total ~ 4.4 MiB

The [2]s are the double buffers from STEP 5. The remaining ~11.5 MiB is left
free ON PURPOSE: the matrix multiplies that run before and after attention
keep an 8 MiB slab of weights resident, and hogging SRAM here would slow
them down. Optimising attention in isolation would be the wrong goal.
"""

# =============================================================================
#  Constants -- where every number comes from
# =============================================================================
S, D, H, G = 2048, 128, 32, 8    # prompt length, head dim, query heads, KV heads
                                 # all four are fixed by the model, not chosen

HPG = H // G                     # = 4. Grouped-Query Attention: 4 query heads
                                 # share one set of K,V. This is why the outer
                                 # loop is over (sequence, GROUP) and the head
                                 # loop sits inside -- converting K,V once and
                                 # reusing it 4 times is a 4x saving.

BR, BC = 128, 256                # query tile, key tile.  BC=256 = array's K limit
                                 # (STEP 3). BR=128 = 4 chunks of 32 = the split
                                 # that balances the arrays (STEP 5).

NQB, NKB = S // BR, S // BC      # = 16 query tiles, 8 key tiles
SHIFT_S, SHIFT_O = 6, 8          # overflow shifts, derived in STEP 4
SCALE = 1.0 / (D ** 0.5)         # the standard 1/sqrt(d) softmax temperature

# Allocated once when the kernel is loaded. Only the CONTENTS change per tile;
# no allocation ever happens inside the loops.
sram = SramMap(  # noqa: F821
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


# =============================================================================
#  The vector-CPU programs used by prefill
#
#  These are pre-compiled and already sitting in the vector CPU. The manager
#  only launches them by name. Each one is memory-bound, so its cost is simply
#      cycles = bytes touched / 128
#  The arithmetic inside is free. This is why all the messy FP32 work
#  (exponentials, rescaling, undoing quantisation) is parked here.
# =============================================================================

def VP_REQUANT_KV(stage, K8, V8T, SK, SV):
    """Convert this group's K and V from 16-bit to 8-bit for the arrays.

    K: one divisor per key row -> K8 + SK.
    V: same idea, BUT also TRANSPOSED on the way out -> V8T + SV.

    Why transpose V? The array computes rows-dot-rows, and step 3 sums over
    keys. So V must be stored as [channel][key], not [key][channel]. Doing it
    here, once, is free-ish; doing it per tile would not be.

    Cost: read 1 MiB + write 512 KiB = 12 288 cycles.

    Run ONCE per (sequence, group), then reused by 4 heads x 2048 queries.
    That amortisation is the whole reason this call sits outside the head loop.
    """


def VP_REQUANT_Q(stage, Q8, SQ):
    """Same conversion for one head's queries. One divisor per query row.
    Cost: read 512 KiB + write 256 KiB = 6 144 cycles. Once per head."""


def VP_SOFTMAX_SUBTILE(S_sub, P8_sub, SQ, SK, M_L, qi, kj, sub):
    """The handyman's main job: turn 32x256 raw scores into 32x256 probabilities.

    Five things happen, in order:

      1. UNDO the quantisation, in FP32:
             s = S_raw * 2^6 * q_divisor * k_divisor * SCALE

      2. MASK, but only on tiles that straddle the diagonal:
             s[i][j] = -inf   where key j comes after query i
         (-inf because exp(-inf) = 0, so masked keys get zero probability.)

      3. UPDATE the running maximum (STEP 7):
             m_new = max(m_old, rowmax(s))
             alpha = exp(m_old - m_new)      <-- stashed in M_L for later

      4. EXPONENTIATE and update the running total:
             p = exp(s - m_new)
             l = alpha * l + rowsum(p)

      5. QUANTISE p back to 8-bit for the array. p is in (0,1] so a fixed
         scale works: P8 = round(p * 255).

    Cost: read 16 KiB + write 8 KiB = 192 cycles. Tiny compared to the
    2048 cycles the array spent producing this tile -- which is exactly why
    the vector CPU can keep up while only being 33% busy.
    """


def VP_ACCUMULATE_SUBTILE(OP_sub, O_ACC, M_L, SV, kj, sub):
    """Fold one partial output tile into the running accumulator.

        O_ACC = alpha * O_ACC  +  OP_raw * 2^8 / 255 * v_divisor

    Two things happen in one pass, and that is the point:
      - the correction factor alpha from STEP 7 is applied
      - the new tile's contribution is added

    We had to read and write O_ACC anyway to do the addition, so multiplying
    by alpha on the way through is free. Online softmax costs no extra traffic.

    O_ACC is FP32, not int16: it accumulates over up to 8 tiles and rounding
    error would build up otherwise.

    Cost: read 8 KiB + read-modify-write 2 x 16 KiB = 320 cycles.
    """


def VP_FINALIZE(O_ACC, M_L, O_OUT):
    """The one division softmax needs, done once at the very end:
           O = O_ACC / l
    then convert FP32 -> int16 (the model's activation format).
    Cost: read 64 KiB + write 32 KiB = 768 cycles."""


# =============================================================================
#  PART A -- THE PREFILL KERNEL
#
#  Loop structure, outermost to innermost:
#
#    (sequence b, KV group g)      128 pairs -- convert K,V here, reuse 4x
#      head h in the group          4 heads -- convert Q here
#        query tile qi             16 tiles of 128 queries
#          key tile kj             qi//2+1 tiles (causal), 256 keys each
#            chunk sub             4 chunks of 32 queries -> the array ops
#
#  Read it as: "for each block of queries, sweep across the keys, carrying a
#  running softmax". The rest is buffering and prefetching so nobody idles.
# =============================================================================

def flash_attention_prefill(Q_dram, K_dram, V_dram, O_dram, B=16):
    """Q_dram[b][h] : [2048 x 128] int16, queries, RoPE already applied
       K_dram[b][g] : [2048 x 128] int16, keys        (per GROUP, not per head)
       V_dram[b][g] : [2048 x 128] int16, values
       O_dram[b][h] : [2048 x 128] int16, where the answer goes
       B            : batch size, 16 sequences

       Total time: 226 ms per layer, arrays ~89% utilised."""

    # ---- Prologue -------------------------------------------------------------
    # Kick off the very first loads before entering the loop. Without this the
    # first iteration would sit and wait for the truck. Every later iteration
    # gets its data prefetched by the iteration before it.
    kv_ev = load_kv(b=0, g=0, buf=0)
    q_ev = load_q(b=0, h=0, buf=0)

    # =========================================================================
    # LEVEL 1: for each (sequence, KV group)
    # =========================================================================
    for n, (b, g) in enumerate(product(range(B), range(G))):
        kv_buf = n % 2            # alternate between the two staging buffers

        # --- Prefetch the NEXT group into the OTHER buffer ---------------------
        # The truck is only 5% busy, so it can run far ahead for free. By the
        # time this group is done computing, the next group has already landed.
        if (b, g) != (B - 1, G - 1):
            nb, ng = divmod(n + 1, G)
            next_kv_ev = load_kv(nb, ng, buf=1 - kv_buf)

        # --- Convert K,V to 8-bit (and transpose V) ---------------------------
        # 12 288 cycles, paid once here and amortised over the 4 heads below.
        # `after=kv_ev` means: wait for the truck to finish unloading first.
        rq_kv = vcpu.run(VP_REQUANT_KV, stage=sram.KV_STAGE[kv_buf], K8=sram.K8,
                         V8T=sram.V8T, SK=sram.SK, SV=sram.SV, after=kv_ev)

        # =====================================================================
        # LEVEL 2: for each of the 4 heads sharing this group's K,V
        # =====================================================================
        for hh in range(HPG):
            h = g * HPG + hh
            q_buf = hh % 2

            # --- Prefetch the NEXT head's Q ----------------------------------
            # Two cases. Normally: the next head in this group.
            # But on the LAST head, prefetch the first head of the NEXT group
            # instead -- otherwise there would be a stall at every group
            # boundary, 128 times per layer.
            if hh + 1 < HPG:
                next_q_ev = load_q(b, h + 1, buf=1 - q_buf)
            elif (b, g) != (B - 1, G - 1):
                next_q_ev = load_q(nb, ng * HPG, buf=1 - q_buf)

            rq_q = vcpu.run(VP_REQUANT_Q, stage=sram.Q_STAGE[q_buf],
                            Q8=sram.Q8[q_buf], SQ=sram.SQ[q_buf], after=q_ev)

            # =================================================================
            # LEVEL 3: for each tile of 128 queries
            # =================================================================
            for qi in range(NQB):

                # Causality (STEP 6): how many key tiles does this query tile
                # actually need? Everything beyond is in the future -> skip.
                n_kb = qi // 2 + 1

                # Reset the running softmax state for this fresh block of
                # queries: m = -inf, l = 0, O_ACC = 0.
                vcpu.run("VP_ZERO", sram.O_ACC, sram.M_L)

                prev_pv = []
                # =============================================================
                # LEVEL 4: sweep across the key tiles.
                #
                # This is the software pipeline of STEP 5. In one pass of this
                # loop, three different things are in flight at once:
                #     scores    for tile kj
                #     softmax   for tile kj
                #     output    for tile kj-1     <-- one behind
                # =============================================================
                for kj in range(n_kb):
                    # Pick this tile's half of each double buffer. The other
                    # half still holds tile kj-1, which is still being used.
                    sb, pb, ob = (sram.S_BUF[kj % 2], sram.P8_BUF[kj % 2],
                                  sram.OP_BUF[kj % 2])

                    # ---- STAGE 1: scores for tile kj ------------------------
                    # 4 chunks of 32 queries. Chunks 0,1,2 -> sa16.
                    # Chunk 3 -> sa32. That 3/1 split is the load balance
                    # from STEP 5. Each chunk costs 2048 cycles either way.
                    s_ev = [None] * 4
                    for sub in range(3):
                        s_ev[sub] = issue_S_subtile_sa16(qi, kj, sub, sb, q_buf,
                                                         after=[rq_q, rq_kv])
                    s_ev[3] = issue_S_subtile_sa32(qi, kj, 3, sb, q_buf,
                                                   after=[rq_q, rq_kv])

                    # ---- STAGE 3, ONE TILE BEHIND: output for tile kj-1 -----
                    # THIS IS THE CORE TRICK. Note `kj - 1` everywhere.
                    # The probabilities for tile kj-1 were produced by the
                    # vector CPU during the previous pass, and are still
                    # sitting in the other half of P8_BUF. Multiply them by V
                    # now, on sa32, filling the time sa16 spends on scores.
                    #
                    # `after=[soft_ev_prev[sub], s_ev[3]]` means: wait for the
                    # probabilities to exist, AND for sa32's own score work to
                    # be queued ahead of it (the array is FIFO).
                    pv_ev = []
                    if kj > 0:
                        pb_prev = sram.P8_BUF[(kj - 1) % 2]
                        ob_prev = sram.OP_BUF[(kj - 1) % 2]
                        for sub in range(4):
                            pv_ev.append(issue_PV_subtile_sa32(
                                kj - 1, sub, pb_prev, ob_prev,
                                after=[soft_ev_prev[sub], s_ev[3]]))

                    # ---- STAGE 2: softmax for tile kj (vector CPU) ----------
                    # Note the order: 3, 0, 1, 2 -- NOT 0, 1, 2, 3.
                    # Chunk 3 went to sa32 and comes back first, so it is
                    # queued first. Queue work in the order results ARRIVE,
                    # not in index order.
                    soft_ev = [None] * 4
                    for sub in (3, 0, 1, 2):
                        soft_ev[sub] = vcpu.run(
                            VP_SOFTMAX_SUBTILE,
                            S_sub=sb + sub * 32 * BC * 2,
                            P8_sub=pb + sub * 32 * BC,
                            SQ=sram.SQ[q_buf], SK=sram.SK, M_L=sram.M_L,
                            qi=qi, kj=kj, sub=sub, after=s_ev[sub])

                    # ---- Fold tile kj-1's output into the accumulator -------
                    # Applies alpha and adds the contribution in one pass
                    # (STEP 7). Again `kj - 1`: still one tile behind.
                    for sub in range(4):
                        if kj > 0:
                            vcpu.run(VP_ACCUMULATE_SUBTILE,
                                     OP_sub=ob_prev + sub * 32 * D * 2,
                                     O_ACC=sram.O_ACC, M_L=sram.M_L,
                                     SV=sram.SV, kj=kj - 1, sub=sub,
                                     after=pv_ev[sub])

                    # Hand this tile's probability events to the next pass,
                    # which will consume them as "the previous tile".
                    soft_ev_prev = soft_ev

                # ---- DRAIN the pipeline ---------------------------------
                # Because output always ran one tile behind, the LAST key
                # tile's output was never issued inside the loop. Do it now.
                # (Nothing overlaps it -- a small, unavoidable bubble at the
                # end of each query tile.)
                pb_last = sram.P8_BUF[(n_kb - 1) % 2]
                ob_last = sram.OP_BUF[(n_kb - 1) % 2]
                last_acc = []
                for sub in range(4):
                    pv = issue_PV_subtile_sa32(n_kb - 1, sub, pb_last, ob_last,
                                               after=[soft_ev_prev[sub]])
                    last_acc.append(vcpu.run(
                        VP_ACCUMULATE_SUBTILE, OP_sub=ob_last + sub * 32 * D * 2,
                        O_ACC=sram.O_ACC, M_L=sram.M_L, SV=sram.SV,
                        kj=n_kb - 1, sub=sub, after=pv))

                # ---- The one division, then ship it out ------------------
                fin = vcpu.run(VP_FINALIZE, O_ACC=sram.O_ACC, M_L=sram.M_L,
                               O_OUT=sram.O_OUT[qi % 2], after=last_acc)
                # O_OUT is double-buffered too, so the next query tile can
                # start writing while the truck is still hauling this one away.
                dma.copy(dst=O_dram[b][h] + qi * BR * D * 2,
                         src=sram.O_OUT[qi % 2],
                         nbytes=BR * D * 2, after=fin)

            # The Q we prefetched at the top of this head becomes the current Q.
            q_ev = next_q_ev
        kv_ev = next_kv_ev

    # The ONLY real "stop and wait" in the entire kernel. Everything above was
    # just the manager queueing jobs and walking away.
    wait_all()


# =============================================================================
#  Helpers -- the actual array commands
#
#  These translate "compute a 32x256 chunk" into the fixed-size ops the
#  hardware understands. Remember the array's shape is wired in: it always
#  produces 16 output columns, so wide outputs must be built from several ops.
# =============================================================================

def load_kv(b, g, buf):
    """Fetch one group's keys and values: 2 x 512 KiB.
    These are stored token-major ([token][channel]), so each token is 256
    contiguous bytes and the whole thing is one flat contiguous read -- the
    fastest possible shape for the truck.

    (Token-major is also what makes decode's per-token append a single
    contiguous 256-byte write. The layout serves both phases.)"""
    e1 = dma.copy(dst=sram.KV_STAGE[buf], src=K_dram[b][g], nbytes=S * D * 2)
    e2 = dma.copy(dst=sram.KV_STAGE[buf] + S * D * 2, src=V_dram[b][g],
                  nbytes=S * D * 2)
    return join(e1, e2)


def load_q(b, h, buf):
    """Fetch one head's queries: 512 KiB, contiguous."""
    return dma.copy(dst=sram.Q_STAGE[buf], src=Q_dram[b][h], nbytes=S * D * 2)


def issue_S_subtile_sa16(qi, kj, sub, s_buf, q_buf, after):
    """Scores for 32 queries x 256 keys, on the 16x16 array.

    The array does 16 queries x 16 keys per op. We need 32 x 256. So:
        32 queries / 16 = 2      ->  outer loop `r`
        256 keys   / 16 = 16     ->  inner loop `kt`
        = 32 ops x 64 cycles = 2048 cycles.

    Each op: 16 query rows (128 bytes each) dotted with 16 key rows
    (128 bytes each), K=128, giving a 16x16 int16 patch of the score matrix.
    `c_stride` tells it the output rows are BC*2 bytes apart, so the patches
    land in the right places inside the 128x256 score buffer."""
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
    """The SAME 32x256 chunk of scores, on the 32x16 array.

    Here all 32 queries go in one op, so only the 16 key-groups need looping:
        16 ops x 128 cycles = 2048 cycles.

    Same total as sa16 -- twice the rows per op, but each op costs twice as
    much, because at K=128 sa32 is limited by how fast it can WRITE results,
    not by the multiplying. This is precisely why sa32 is not simply "the
    fast array": on scores it is a tie. It only pulls ahead on the output
    step, where K=256 (see the next function)."""
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
    """Step 3: probabilities x values, for 32 queries -> 32 x 128 outputs.

        O_partial[32 queries x 128 channels]
            = P8[32 x 256 keys] . V8T[128 channels x 256 keys]^T

    Note what is being summed over: KEYS, all 256 of them. So K=256 here,
    the array's maximum. That is the whole reason BC was chosen as 256.

    The output is 128 channels wide but the array emits 16 columns per op:
        128 / 16 = 8 ops x 128 cycles = 1024 cycles.

    And HERE is where sa32 earns its place: at K=256 it processes 32 rows in
    the same 128 cycles that sa16 needs for 16. Twice the throughput. So all
    four output chunks go to sa32, and sa16 is left doing scores. That is
    the 3/1 split from STEP 5.

    `B=sram.V8T + (ct*16)*S + kj*BC` -- V8T is stored [channel][key], so
    stepping a channel means stepping S=2048 bytes, and kj*BC picks out this
    tile's 256 keys. That layout is what VP_REQUANT_KV built the transpose for."""
    evs = []
    for ct in range(D // 16):
        evs.append(sa32.matmul(
            A=p8_buf + sub * 32 * BC,
            B=sram.V8T + (ct * 16) * S + kj * BC,
            C=op_buf + sub * 32 * D * 2 + ct * 16 * 2,
            K=BC, out_shift=SHIFT_O, c_stride=D * 2, after=after))
    return join(*evs)


# =============================================================================
#  PART B -- THE DECODE KERNEL
#
#  =============  WHY THIS IS A COMPLETELY DIFFERENT PROGRAM  =============
#
#  Now we generate ONE new word. Per (sequence, group), per layer:
#      queries : 4     (the 4 heads of the group, one token each)
#      keys    : L = 2049 .. 2304   (the whole conversation so far)
#      math    : 2 x 4 x L x 128       ~ 2.2 M MACs
#      data    : L x 2 x 128 x 2 bytes ~ 1.1 MB
#      ratio   : ~2 MAC per byte
#
#  Compare to prefill's 460. The chip breaks even at 24. We are far, FAR on
#  the memory-bound side. Every key and value is read from DRAM, used once,
#  and thrown away -- there is no reuse to exploit, because there is only
#  one query.
#
#  The floor: 1.1 MB at 64 bytes/cycle = 8L cycles = 17.4k cycles per
#  (sequence, group). NOTHING can beat that. The only question is what runs
#  underneath it without falling behind.
#
#  Option 1 -- use the systolic arrays.  REJECTED.
#      * The array processes 16 or 32 rows per op. We have 4 useful rows.
#        So at best 25% of the input port does real work.
#      * K and V would have to be converted to 8-bit and V transposed EVERY
#        step, because they are used exactly once and then discarded.
#        That is 1.6 MB of vector-CPU traffic per (sequence, group) -- work
#        that only exists to feed the arrays.
#      * Arithmetic: 136 score ops x 64 + 68 output ops x 128 = 17.4k cycles.
#        EXACTLY the DMA time. So the arrays would be the co-bottleneck, for
#        more complexity and worse numerics.
#
#  Option 2 -- use the vector CPU.  CHOSEN.
#      * Reads every byte once through the chip's widest port (128 B/cycle)
#        = 4L cycles = HALF the DMA time. Comfortably ahead, never the limit.
#      * Works in FP32: no conversion, no transpose, no precision loss.
#      * Leaves both arrays completely free for the matrix multiplies that
#        surround attention.
#
#  So the design goal inverts. In prefill: never let an array idle.
#  In decode: NEVER LET THE DRAM PIPE RUN DRY.
#
#  The structure that achieves it: a conveyor belt. Chop the cached keys and
#  values into fixed chunks, stream them through a ring of 4 SRAM buffers,
#  and keep the truck ~3 chunks ahead of the handyman at all times.
# =============================================================================

CH = 256      # keys per chunk. K chunk 64 KiB + V chunk 64 KiB = 128 KiB.
              #     truck : 128 KiB / 64 B  = 2048 cycles   <- the pace-setter
              #     vcpu  : 128 KiB / 128 B = 1024 cycles   (50% busy)
              # The 2x headroom is deliberate: the consumer must never become
              # the bottleneck, or the pipe stalls.

NRING = 4     # buffers in the ring. 3 in flight + 1 being filled. Enough that
              # a hiccup on either side does not propagate into a stall.

sram_dec = SramMap(  # noqa: F821
    KV_RING=[alloc(2 * CH * D * 2) for _ in range(NRING)],   # 4 x 128 KiB
    ACC=alloc(HPG * (D + 2) * 4),   # m, l, and o[128] per head, FP32. 2 KiB.
)
# Total decode footprint is only ~0.83 MiB (vs 4.4 MiB in prefill). The other
# ~15 MiB goes to the surrounding matrix multiplies, which in decode are ALSO
# memory-bound and need every byte of buffering they can get. Attention giving
# SRAM back is part of the design, not an accident.


def VP_DEC_CHUNK(q4, Kc, Vc, n_keys, ACC, first):
    """Process one 256-key chunk for the 4 heads of a group. All FP32.

    This is the entire attention computation -- all three steps -- done by
    the handyman alone, with no array involved and no quantisation:

        s[h][j] = q4[h] . Kc[j] * SCALE            scores (4 x n_keys)
        m_new   = max(m[h], max_j s[h][j])         running max  (STEP 7)
        alpha   = exp(m[h] - m_new)                correction factor
        p[h][j] = exp(s[h][j] - m_new)             probabilities
        l[h]    = alpha * l[h] + sum_j p           running total
        o[h]    = alpha * o[h] + sum_j p[h][j] * Vc[j]      running output

    Same online softmax as prefill. Only the executor changed.

    Cost: read Kc + Vc = 128 KiB -> 1024 cycles. q4 and ACC are tiny enough
    to live in registers, so they cost nothing. Half the truck's 2048 cycles,
    exactly as intended."""


def VP_DEC_FINALIZE(q4, k_new, v_new, ACC, o_out):
    """Two small jobs at the end of each (sequence, group):

    1. Fold in the NEW token's own key and value. These come straight from
       SRAM -- the matrix multiply earlier in this same layer just produced
       them. We do NOT read them back from the DRAM cache. That saves a
       round-trip and, more importantly, removes a write-then-read hazard on
       the cache (the write may still be in flight).

    2. Divide by l and convert to int16:  o_out[h] = int16(o[h] / l[h])

    Cost: ~3 KiB. Negligible."""


def flash_attention_decode(Q_ALL, KV_NEW, K_dram, V_dram, ctx, O_ALL, B=16):
    """Q_ALL[b][h]  : [128] int16, the new token's query (from this layer's
                      QKV matrix multiply -- already in SRAM)
       KV_NEW[b][g] : the new token's k and v, likewise already in SRAM
       ctx[b]       : how many tokens are already cached for sequence b
                      (2048 .. 2303 -- sequences are at different lengths)
       O_ALL[b][h]  : [128] int16 result, left in SRAM for the next matrix
                      multiply to pick up. Never goes to DRAM.

       Time: n_chunks x 2048 cycles per (sequence, group), DMA-bound.
       Per layer: 128 pairs x ~8.5 chunks x 2048 = 2.2 M cycles = 2.2 ms."""

    # The ring hands out buffers in strict rotation. Asking for a buffer also
    # returns "the event that says the handyman finished with it last time",
    # so a refill can be queued immediately and will simply wait its turn.
    ring = RingScheduler(NRING)

    # Flatten the whole layer's work into one list of chunks, in the exact
    # order it will be streamed. Note this crosses sequence and group
    # boundaries -- the belt does not stop and restart between them.
    plan = [(b, g, c)
            for b in range(B)
            for g in range(G)
            for c in range(ceil_div(ctx[b], CH))]

    # -------------------------------------------------------------------------
    # THE TRUCK: queue the ENTIRE layer's transfers up front.
    #
    # Not chunk-by-chunk on demand -- all of it, right now. The DMA queue is
    # deep, so the hardware just works through it back to back and the pipe
    # never has a gap waiting for the manager to think.
    #
    # The only thing throttling this is `after=free_ev`: a buffer cannot be
    # refilled until the handyman has finished reading its previous contents.
    # With 4 buffers that means the truck naturally settles ~3 chunks ahead.
    # -------------------------------------------------------------------------
    land = {}
    for (b, g, c) in plan:
        buf, free_ev = ring.next()
        n_keys = min(CH, ctx[b] - c * CH)     # last chunk may be partial
        e1 = dma.copy(dst=sram_dec.KV_RING[buf],
                      src=K_dram[b][g] + c * CH * D * 2,
                      nbytes=n_keys * D * 2, after=free_ev)
        e2 = dma.copy(dst=sram_dec.KV_RING[buf] + CH * D * 2,
                      src=V_dram[b][g] + c * CH * D * 2,
                      nbytes=n_keys * D * 2, after=free_ev)
        land[(b, g, c)] = (buf, n_keys, join(e1, e2))

    # -------------------------------------------------------------------------
    # THE HANDYMAN: consume the chunks in the same order.
    #
    # Each chunk waits for its own arrival (`after=landed`) and nothing else.
    # Since the truck is 3 chunks ahead, that wait is essentially always
    # already satisfied by the time the handyman gets there.
    # -------------------------------------------------------------------------
    for b in range(B):
        for g in range(G):
            q4 = Q_ALL[b][g * HPG:(g + 1) * HPG]    # the group's 4 heads
            n_ch = ceil_div(ctx[b], CH)
            done = None
            for c in range(n_ch):
                buf, n_keys, landed = land[(b, g, c)]
                done = vcpu.run(VP_DEC_CHUNK, q4=q4,
                                Kc=sram_dec.KV_RING[buf],
                                Vc=sram_dec.KV_RING[buf] + CH * D * 2,
                                n_keys=n_keys, ACC=sram_dec.ACC,
                                first=(c == 0), after=landed)
                # Hand the buffer back. This `done` event is the `free_ev`
                # that unblocks the refill queued way back in the loop above.
                # This single line is what keeps the belt moving without
                # letting it overrun.
                ring.release(buf, done)

            vcpu.run(VP_DEC_FINALIZE, q4=q4,
                     k_new=KV_NEW[b][g].k, v_new=KV_NEW[b][g].v,
                     ACC=sram_dec.ACC,
                     o_out=O_ALL[b][g * HPG:(g + 1) * HPG], after=done)

    # One last detail: the O-projection matrix multiply that follows attention
    # has its first weight transfer queued right behind the final KV chunk. So
    # the DRAM pipe stays saturated straight through the phase boundary
    # instead of draining and refilling. In a memory-bound phase, that gap
    # would be pure lost time.
    wait_all()


# =============================================================================
#  SUMMARY -- the two kernels in one sentence each
#
#  PREFILL  (compute-bound, 226 ms/layer, arrays are the resource)
#    Tile the problem so the arrays always have work; run the output multiply
#    one key-tile behind the score multiply so both arrays stay busy through
#    softmax; double-buffer everything so nobody blocks on memory; prefetch
#    the next head and group during the current one; skip the ~44% of tiles
#    that causality makes unnecessary.
#
#  DECODE  (memory-bound, 2.2 ms/layer, the DRAM pipe is the resource)
#    Give up on the arrays entirely -- with one query they would waste 75% of
#    their input port and demand pointless conversion work. Instead stream
#    the KV cache through a 4-buffer ring straight into the vector CPU, which
#    keeps up at half the DMA rate, and queue the whole layer's transfers in
#    advance so the pipe never has a gap.
#
#  The shared idea: find the one resource that is actually the bottleneck,
#  then arrange everything else -- layout, buffering, work assignment, even
#  which engine runs the math -- so that resource never stops.
# =============================================================================
