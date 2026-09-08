# Section 2.2, explained from scratch

*A plain-language companion to `../C2/report.md`. Same machine, same numbers, same answers —
but every step is derived rather than asserted, and nothing assumes you already know
the vocabulary.*

`../C2/report.md` is the submission: dense, quotes each problem statement, states results.
**This file is the reasoning behind it.** Read this one to understand *why* each number
is what it is; read that one for the compact answer.

Companion files: `flash_attention_annotated.py` (the kernel, explained the same way)
and `../C2/perf_model.py` (the code that produces every number here).

---

## Contents

- [Part 0 — What is this machine?](#part-0--what-is-this-machine)
- [Part 1 — The one number that explains everything: 24](#part-1--the-one-number-that-explains-everything-24)
- [Part 2 — Why LLM inference has two personalities](#part-2--why-llm-inference-has-two-personalities)
- [Problem 1 — The flash-attention kernel](#problem-1--the-flash-attention-kernel)
- [Problem 2 — What is the bottleneck?](#problem-2--what-is-the-bottleneck)
- [Problem 3 — How fast is it really?](#problem-3--how-fast-is-it-really)
- [Problem 4 — Two ways to make it better](#problem-4--two-ways-to-make-it-better)
- [Problem 5 — Four accelerators](#problem-5--four-accelerators)
- [Appendix — Assumptions, and why each one matters](#appendix--assumptions-and-why-each-one-matters)

---

# Part 0 — What is this machine?

Figure 1 is a block diagram. Boxes are components, arrows are wires. For performance
analysis, forget the boxes: **only the arrows matter**.

Think of it as a factory. Every arrow is a conveyor belt with a fixed width. A job takes
as long as its slowest belt — everything else is just waiting.

The clock is 1 GHz, which makes the arithmetic easy:

> **1 cycle = 1 nanosecond.**
> A belt described as "512 bit/cycle" carries 512/8 = **64 bytes per cycle** = 64 GB/s.

## 0.1 The belts

| Wire | Given as | Bytes/cycle | What it means in practice |
|---|---|---|---|
| DRAM ↔ DMA engine | 512 bit/cyc | 64 | |
| DMA engine ↔ SRAM | 512 bit/cyc | 64 | **in series with the one above** → the end-to-end path is 64 GB/s, and every weight and every KV byte must cross it |
| SRAM ↔ vector CPU | 1024 bit/cyc | **128** | the widest belt on the chip |
| SRAM → 16×16 array (in) | 512 bit/cyc | 64 | |
| 16×16 array → SRAM (out) | 64 bit/cyc | **8** | ← notice how small |
| SRAM → 32×16 array (in) | 768 bit/cyc | 96 | |
| 32×16 array → SRAM (out) | 64 bit/cyc | **8** | ← and again |
| Host ↔ DRAM | — | — | the only door to the outside world |
| Control CPU → engines | 200/300/100 cyc | — | not a belt at all: pure command latency |

Two entries deserve a second look.

**The output ports are 8 bytes/cycle.** That is *eight times narrower* than the input
port feeding the same array. An array can take data in far faster than it can push
results out. Keep this in mind — it turns out to be the answer to Problem 2.

**The two DRAM links are in series.** Data goes DRAM → DMA → SRAM, crossing both.
Widening one alone changes nothing; the other immediately becomes the limit. This is
also why Problem 2's elasticity table shows −0.55 for a single link but −1.01 for both.

## 0.2 Six consequences

**(a) The compute units never see DRAM.**
Only the DMA engine reaches DRAM. Only SRAM feeds the arrays and the vector CPU.
So *every* kernel has the shape:

```
DMA in  →  compute  →  DMA out
```

and if you don't want the compute waiting for the DMA, you need two of every buffer:
fill one while the other is being consumed. **This single fact is why the word
"double-buffered" appears everywhere in Problem 1.**

**(b) The arrays are limited by their ports, not by their multipliers.**

A "tile op" streams two int8 operands in and one int16 result out. It is pipelined —
the next op's inputs flow in while this op's outputs flow out — so:

> cost of one op = max(input bytes ÷ input bandwidth, output bytes ÷ output bandwidth)

Let's actually compute this. `K` is the contraction depth (how many terms each dot
product sums).

**16×16 array, K=256:**
- inputs: 16 rows of A + 16 rows of B, each 256 bytes = (16+16)×256 = 8192 B, at 64 B/cyc → **128 cycles**
- outputs: 16×16 results × 2 bytes = 512 B, at 8 B/cyc → **64 cycles**
- cost = max(128, 64) = **128 cycles**, and it did 16×16×256 = 65 536 MACs → **512 MAC/cycle**

**32×16 array, K=256:**
- inputs: (32+16)×256 = 12 288 B at 96 B/cyc → **128 cycles**
- outputs: 32×16×2 = 1024 B at 8 B/cyc → **128 cycles**
- cost = **128 cycles**, doing 32×16×256 = 131 072 MACs → **1024 MAC/cycle**

Now halve K to 128, which is the head dimension — the depth of Q·Kᵀ:

**16×16, K=128:** inputs halve to 64 cycles, outputs stay 64 (output size doesn't depend
on K!) → cost 64 → still **512 MAC/cycle**. Perfectly balanced.

**32×16, K=128:** inputs halve to 64 cycles, outputs stay **128** → cost 128 →
**512 MAC/cycle**. The array spends half its time waiting on its own output port.

| Array | K | in | out | cost | MAC/cycle | |
|---|---|---|---|---|---|---|
| 16×16 | 256 | 128 | 64 | **128** | **512** | input-bound |
| 16×16 | 128 | 64 | 64 | 64 | 512 | balanced |
| 32×16 | 256 | 128 | 128 | **128** | **1024** | balanced |
| 32×16 | 128 | 64 | **128** | 128 | **512** | **output-bound — half rate** |

**Read the last row again.** At K=128 the big array is no faster than the small one.
The constants in Figure 1 were clearly chosen so that everything balances at K=256 —
but the head dimension of the model is 128, and that is exactly where the 32×16 array
loses half its value.

Peak for the chip: 512 + 1024 = **1536 MAC/cycle = 3.07 TOPS**.

**(c) The arrays are hungrier than DRAM can feed them.**
The two arrays together can ingest weights at 32+64 = 96 B/cycle. DRAM delivers 64.
So on any workload where weights are read once and used once, the arrays starve.
That is decode. (Problems 3 and 4.)

**(d) One ratio governs everything.** Divide peak compute by memory bandwidth:

> 1536 MAC/cycle ÷ 64 B/cycle = **24 MAC per DRAM byte**

This is the *ridge point*. It is the break-even arithmetic intensity of the machine:

- A kernel doing **fewer** than 24 MACs per byte it pulls from DRAM → **memory-bound**.
  Its time is `bytes ÷ 64 GB/s`. Adding compute wouldn't help.
- A kernel doing **more** than 24 → **compute-bound**.
  Its time is `MACs ÷ 1.536·10¹² per second`. Adding bandwidth wouldn't help.

`../C2/roofline.png` (Figure 2) is exactly this: a sloped line (memory limit) meeting a flat
line (compute limit) at 24, with the workload's four phases plotted as dots.

Every single answer below is, at bottom, "which side of 24 is this?"

**(e) The control CPU is on no data path.**
It issues commands with 200/300/100-cycle latencies. If it waited for each one, a
128-cycle tile op preceded by a 100-cycle command wait would run at 128/228 = **56%**
efficiency. So commands must be *queued* — fired off asynchronously, latency paid once
per burst instead of once per op. This is assumption **A7**, and Problem 2 shows it
is the difference between 226 ms and 476 ms per layer.

**(f) The four accelerators can only talk through the host.** No direct links.
(Problem 5.)

---

# Part 1 — The one number that explains everything: 24

Before going further, internalise the ridge.

**Arithmetic intensity** = MACs performed ÷ bytes fetched from DRAM.

It is a property of the *algorithm and its data reuse*, not of the hardware. The
hardware just tells you where the break-even sits — here, 24.

Two quick examples from this workload:

*A matrix multiply during prefill.* One weight byte is read, and then used by all
32 768 tokens flowing past it. Intensity ≈ 32 768. Massively compute-bound.

*The same matrix multiply during decode.* One weight byte is read, used by 16 tokens
(one per sequence in the batch), discarded. Intensity = 16. Below 24 → memory-bound.

Same code. Same weights. **The only thing that changed is how many tokens are in
flight**, and that flips the machine from one regime to the other.

That is the entire report in one paragraph.

---

# Part 2 — Why LLM inference has two personalities

Generating a reply happens in two phases, and they are not variations on a theme —
they are opposite problems.

**PREFILL.** The user's 2048-token prompt arrives. All of it is processed at once,
through all 32 layers, and the K/V results are written to the cache. With 16 sequences
in the batch, 16 × 2048 = **32 768 tokens** pass through each layer together.
The time this takes is the **TTFT** — how long the user stares at nothing.

**DECODE.** Now generate token 1. Then token 2. Then 254 more. Each step must read
**every weight in the model** and **the entire KV cache** — to produce 16 tokens.
Step time sets **interactivity** — how fast text appears.

| | **Prefill** → TTFT | **Decode** → interactivity |
|---|---|---|
| tokens through a layer at once | 32 768 | 16 |
| matmul: MAC per weight byte | 32 768 | 16 |
| attention: MAC per KV byte | ≈ 460 | ≈ 2 |
| vs. the ridge of 24 | far above → **compute-bound** | far below → **memory-bound** |
| the binding resource | SRAM ↔ array ports | DRAM → DMA → SRAM, 64 GB/s |
| MACs per layer | 7.5 × 10¹² | 3.8 × 10⁹ |
| DRAM bytes per layer | ≈ 12 GB, but only 4% of the time | 218 MB weights + 143 MB KV = **100% of the time** |
| what to optimise | keep the arrays busy | keep the pipe full |
| what SRAM is for | holding things that get reused | a small landing strip; nothing is reusable |
| arrays | 100% busy | 67% on matmuls, **idle** in attention |
| time per layer | 4.92 s | 5.5 ms |
| result | **TTFT ≈ 157 s** | **189 ms/token → 5.3 tok/s per sequence** |

Look at the "MAC per byte" row: 32 768 vs 16 for the matmuls, 460 vs 2 for attention.
**Three to four orders of magnitude apart**, straddling the ridge. This is why every
problem below gets answered twice.

**Where the workload sizes come from** (Llama 3.1 8B, int8 weights, 16 prompts of
2048 in, 256 out):

- linear weights 6.98 GB + LM head 0.53 GB = **7.50 GB streamed every decode step**
  → 7.50 GB ÷ 64 GB/s = **117 ms just for weights**
- KV cache is 128 KiB per token (32 layers × 8 KV heads × 128 dims × 2 tensors × 2 B)
  → at average context 2176: 16 × 2176 × 128 KiB = **4.56 GB per step = 71 ms**
- prefill linear MACs = 2.29 × 10¹⁴ → ÷ 1.536 × 10¹² = **149 s**
- prefill attention = 9.9 × 10¹² MACs, only 4% of prefill

Notice: **117 + 71 ≈ 189 ms**, which *is* the decode step time. The whole of decode is
those two byte counts divided by 64 GB/s. Nothing else matters.

---

# Problem 1 — The flash-attention kernel

> *"Provide a pseudocode implementation of an optimized flash-attention kernel for this
> architecture, detailing the SRAM allocation strategy, the tiling strategy and the
> overall computation dataflow. Keep in mind that this code, once implemented, would be
> executed on the control CPU."*

The deliverable is `../C2/flash_attention_pseudocode.py` (annotated version:
`flash_attention_annotated.py`). Below is the reasoning; the code file has the
line-by-line walkthrough.

**"Executed on the control CPU" is the key phrase in the question.** The control CPU is
on no data path (0.2e). So the kernel is not code that computes attention — it is code
that *dispatches* attention: it programs DMA descriptors, enqueues tile commands to the
arrays, launches vector programs, and chains them together with events. It never touches
a number.

And because the two phases are on opposite sides of the ridge, one kernel cannot serve
both. There are two entry points.

| | **Part A — prefill** | **Part B — decode** |
|---|---|---|
| shape per (sequence, KV group) | 4 heads × 2048 queries × 2048 keys, causal | 4 heads × **1** query × L keys, L = 2049…2304 |
| MACs / DRAM bytes | 2.4 G / 5 MB → **460 MAC/B** | 2.2 M / 1.1 MB → **2 MAC/B** |
| regime | compute-bound | memory-bound |
| who is busy | both arrays 100%, vector 38%, DMA 5% | DMA 100%, vector 50%, **arrays idle** |
| the goal | keep both arrays saturated | keep the DRAM pipe saturated |
| tiling | 128-query × 256-key blocks, 32-row chunks | 256-key chunks of a stream |
| SRAM | 4.4 MiB | 0.8 MiB |
| time per layer | 226 ms | 2.2 ms |

## 1.A — Prefill (compute-bound)

### What has to happen

For one head: `S = Q·Kᵀ` (2048×2048, summing over d=128), `P = softmax(S)` with a causal
mask, `O = P·V` (summing over keys).

FlashAttention's idea: process `S` block by block, carrying a running max `m` and running
sum `l` per row, so the full `S` is never materialised.

**But note the motive here is unusual.** Normally you tile because S wouldn't fit in
memory. Here one head's S is 8 MiB and would actually fit in the 16 MiB SRAM. The real
reason is **bandwidth**: S leaves the arrays through an 8 B/cycle port and then has to
be read again by the vector CPU. Writing it once and reading it once is already
expensive; writing it, storing it, and re-reading it later would be worse. So S must be
produced and consumed exactly once, immediately. Tiling is what makes that possible.

### Design goals, in priority order

1. **Both arrays busy 100% of the time.** They are the scarce resource — everything
   else has slack.
2. Vector CPU off the critical path.
3. Each K/V byte read from DRAM once per layer, shared by the 4 query heads of its
   GQA group.
4. No control-CPU latency ever stalls an engine.

### Mapping the two matmuls onto two unequal arrays

The arrays compute `C = AᵀB` with both operands laid out as `[rows × K]` — "rows dot
rows". So:

- `S = Q·Kᵀ` : query rows (128 B) against key rows (128 B), **K = 128**
- `O = P·V` : P rows (256 keys) against rows of **Vᵀ** (256 keys), **K = 256**

The second one needs V *transposed*. The vector CPU builds Vᵀ in SRAM once per
(sequence, group) while it is converting V to int8 anyway — 12k cycles, amortised over
4 heads × 2048 queries, so effectively free.

> *Why not just store the cache transposed in DRAM and skip this?* Because decode has to
> append one token per step. In token-major layout that's a single contiguous 256-byte
> write. Transposed, it becomes **128 scattered 2-byte writes**. The layout is chosen
> for decode's sake, and prefill pays a small, amortisable cost to fix it up.

Now the interesting part. From the table in 0.2(b):

- at **K = 128** (that's Q·Kᵀ) both arrays give 512 MAC/cycle — *they are equal*
- at **K = 256** (that's P·V) the 32×16 gives 1024 — *twice as fast*

So the 32×16 array has no advantage on scores but a 2× advantage on outputs. The right
split is not "half each" — it's "give the big array the work only it is good at."

Let the 32×16 do all of P·V plus a fraction *x* of Q·Kᵀ.

Both matmuls have the **same** MAC count — Q·Kᵀ is (queries × keys × 128 dims), P·V is
(queries × 128 dims × keys). Call it W each. Time = MACs ÷ rate, so the two arrays
finish together when:

```
(1 − x)·W/512     =     x·W/512    +    W/1024
16×16 on Q·Kᵀ           32×16 on Q·Kᵀ    32×16 on P·V
   at 512/cyc              at 512/cyc      at 1024/cyc
```

Divide through by W/512:  (1 − x) = x + 0.5  →  **x = 0.25**.

- **16×16 array → 75% of Q·Kᵀ**
- **32×16 array → 25% of Q·Kᵀ + 100% of P·V**

Sustained: **1365 MAC/cycle = 89% of the 1536 peak.**

(For comparison: splitting work proportionally to array speed gives 1229 MAC/cycle,
because the 32×16 sits idle whenever Q·Kᵀ is the only work available. The function
`split_two_arrays` in `../C2/perf_model.py` confirms 0.25 is the optimum.)

### Tiling — and where the numbers come from

| Parameter | Value | Why *this* value |
|---|---|---|
| key block **BC** | **256** | It equals Kmax. P·V contracts over keys, so one tile op can consume an entire key block. Any smaller wastes the array's depth; larger is illegal. |
| query block **BR** | **128** = 4 chunks of 32 | 32 = the 32×16 array's row count. Four chunks is what makes the split above land exactly: 3 score chunks on the 16×16 = 1 score + 4 output chunks on the 32×16. |
| S chunk (32q × 256k) | 32 ops × 64 cyc on 16×16, **or** 16 ops × 128 cyc on 32×16 | **2048 cycles either way** — this is the K=128 tie |
| PV chunk (32q × 128d) | 8 ops × 128 cyc on 32×16 | 1024 cycles — here the 32×16 is 2× ahead |
| one 128×256 block | **6144 cycles on each array** | 16×16: 3 × 2048. 32×16: 2048 + 4 × 1024. **Equal.** |
| vector CPU per block | 256 KiB → 2048 cycles | 33% of the block — comfortable slack |
| causal skipping | **72 of 128** blocks per head | above-diagonal blocks never issued; diagonal blocks computed in full and masked by the vector CPU |

That "6144 on each array" row is the design landing. Neither array ever waits for the
other.

### SRAM allocation

The strategy is two rules:

1. **Anything reused across many tiles stays resident** for as long as it's reused.
2. **Anything handed from one engine to another is doubled**, so producer and consumer
   never block each other.

| Region | Size | filled by → drained by | lives for |
|---|---|---|---|
| KV staging ×2 (16-bit) | 2 × 1 MiB | DMA → vector | one (sequence, group) |
| K8, Vᵀ8, scales | 524 KiB | vector → arrays | one group, **shared by 4 heads** |
| Q staging ×2 (16-bit) | 2 × 512 KiB | DMA → vector | one head |
| Q8 ×2 + scales | 2 × 264 KiB | vector → arrays | one head |
| S ×2 int16 | 2 × 64 KiB | arrays → vector | one key block |
| P8 ×2 uint8 | 2 × 32 KiB | vector → 32×16 | one key block |
| O partial ×2 int16 | 2 × 32 KiB | 32×16 → vector | one key block |
| O accumulator fp32 + m/l/α | 66 KiB | vector (read-modify-write) | one query block |
| O out ×2 int16 | 2 × 32 KiB | vector → DMA | one query block |
| **Total** | **≈ 4.4 MiB of 16 MiB** | | |

Every `×2` is rule 2. The "lives for" column is rule 1 — and note how the lifetimes
nest exactly like the loop nest.

**The 11.5 MiB left over is deliberate, not laziness.** The matmuls before and after
attention keep an 8 MiB weight slab resident (A9). Attention grabbing that SRAM to
optimise itself would slow down phases that take 20× longer. *Optimising a component
in isolation is the wrong objective.*

### Dataflow

Loop order: **(sequence b, KV group g) → head h → query block qi → key block kj → chunk.**

K/V are loaded and converted once per (b,g); Q once per head; the next group's K/V and
next head's Q are prefetched during compute — easy, since the DMA is 5% busy.

Steady state, software-pipelined by one key block:

```
cycle      0          2048         4096         6144
16×16   | S(chunk0,j)| S(chunk1,j)| S(chunk2,j)|
32×16   | S(chunk3,j)| PV(chunk0,j−1) PV(chunk1,j−1) PV(chunk2,j−1) PV(chunk3,j−1) |
vector  |  softmax(chunk3,j) softmax(chunk0..2,j)  accumulate(chunk0..3, j−1)      |  33%
DMA     |  prefetch next Q / next K,V ;  write back finished O blocks              |   5%
```

**The `j−1` on the 32×16 row is the whole trick.** While the small array computes scores
for block *j*, the big array computes outputs for block *j−1*. Without that one-block
lag, the arrays would idle during every softmax.

Per layer: 16 sequences × 8 groups × 4 heads × 72 blocks × 6144 cycles = **226 ms**,
with arrays at 100%, vector CPU at 38%, DRAM at 5%.

### Numerics

The arrays eat int8 and emit int16; the model uses 16-bit activations. So the vector CPU
converts on the way in and un-converts on the way out.

- **In:** Q, K per row; V per (channel, 256-key block); P as uint8 with fixed scale 1/255
  (probabilities are already in (0,1], so no search needed).
- **Overflow:** right-shift by 6 after Q·Kᵀ (max |sum| = 127·127·128 = 2.06e6, too big for
  int16) and by 8 after P·V (255·127·256 = 8.29e6).
- **Out:** the vector CPU undoes shifts and scales in fp32, applies 1/√d, the mask, exp,
  the running max/sum, and accumulates O in fp32.
- The online-softmax rescale α = exp(m_old − m_new) is applied *when the P·V partial of
  the same block is folded in* — a pass that was happening anyway. **So online softmax
  costs zero extra traffic.**

## 1.B — Decode (memory-bound)

### Why it must be a different program

Per (sequence, group) per layer: 4 query vectors against L ≈ 2176 keys and values.
2.2 M MACs, but 1.1 MB of K/V read from DRAM **once and used once**.

The floor is immediate: 1.1 MB ÷ 64 B/cycle = 8L cycles = 17.4k cycles per pair,
**2.2 ms per layer** for the 128 pairs. Nothing can beat that. The only question is
which engine consumes the stream without becoming a *second* bottleneck.

**Option 1 — the systolic arrays. Rejected.**
- Aᵀ must be a 16-row tile, and only 4 rows (the heads) are useful → ≤ 25% of the input
  port does real work.
- K/V would have to be requantised and V transposed **every single step**, since nothing
  is ever reused. That's 1.6 MB of extra vector traffic per pair — work that exists only
  to feed the arrays.
- And the array time works out to 136 ops × 64 + 68 ops × 128 = **17.4k cycles**, exactly
  equal to the DMA time. So you'd have two bottlenecks instead of one, plus a precision
  loss, plus the arrays unavailable for the matmuls.

**Option 2 — the vector CPU. Chosen (A10).**
- Reads each byte once through the chip's widest port: 4L cycles = **50% of the DMA
  time**. Comfortably ahead, never the limit.
- fp32 throughout: no requantisation, no transpose, no precision loss.
- Both arrays stay free for the surrounding matmuls.

**The design objective flips.** Prefill: never let an array idle. Decode: **never let
the DMA queue run dry.**

### Tiling = chunking the stream

CH = 256 keys per chunk → 64 KiB of K + 64 KiB of V = 128 KiB.

- DMA per chunk: 128 KiB ÷ 64 = **2048 cycles** ← the pace-setter
- vector per chunk: 128 KiB ÷ 128 = **1024 cycles** ← 2× headroom, on purpose

Online softmax across chunks for the 4 heads (m, l, o in fp32, 2 KiB total).

The new token's own k and v come **from SRAM**, where this layer's QKV matmul just
produced them — not read back from DRAM. Saves a round trip and, more importantly,
sidesteps a write-then-read hazard on the cache.

### SRAM allocation

| Region | Size | purpose |
|---|---|---|
| KV ring ×4 | 4 × 128 KiB | DMA landing ring, 3 chunks in flight |
| Q_ALL [16 × 4096] int16 | 128 KiB | output of the QKV matmul (already there) |
| KV_NEW | 64 KiB | the new token's k, v (same matmul) |
| m, l, o accumulators fp32 | 2 KiB | current pair |
| O_ALL [16 × 4096] int16 | 128 KiB | input of the O-projection matmul |
| **Total** | **≈ 0.8 MiB** | the other ≈ 15 MiB goes to decode's weight slabs |

**0.8 MiB versus prefill's 4.4 MiB.** Attention giving SRAM back is part of the design:
in decode the *matmuls* are memory-bound too and need every byte of buffering they can
get.

### Dataflow

```
cycle    0        2048      4096      6144      8192 …
DMA    | ch0     | ch1     | ch2     | ch3     | ch4 …   100% busy
vector |         | ch0     | ch1     | ch2     | ch3 …    50% busy
arrays |              idle (the matmuls use them, before and after)     |
```

The control CPU queues the next pair's descriptors while the vector CPU is still on the
current one, so the ring stays three chunks ahead and the DMA never waits. And the first
weight slab of the *following* matmul is queued right behind the last KV chunk — so the
DRAM pipe doesn't drain and refill at the phase boundary. In a memory-bound phase that
gap would be pure lost time.

Per layer: 128 pairs × 8.5 chunks × 2048 cycles = **2.2 ms**, DRAM-bound.

---

# Problem 2 — What is the bottleneck?

> *"Identify and explain the bottleneck of this accelerator's architecture for the
> flash-attention kernel, i.e. the parameter with the largest elasticity with respect to
> time required for this kernel to complete."*

## First: what is "elasticity"?

The question asks for a specific measure, so let's be precise about it.

> **e(p) = (ΔT / T) ÷ (Δp / p)**

"If I change parameter *p* by 1%, by what percentage does the runtime T change?"

Reading the values:

- **e = −1** → time is *inversely proportional* to p. Double p, halve the time.
  **This parameter is the bottleneck.**
- **e = 0** → p is irrelevant. There is slack; changing it does nothing.
- **e = −0.5** → p matters, but shares the blame with something else.

Why this is the right question to ask: "what's the bottleneck" is vague when several
resources are partly busy. Elasticity answers it operationally — *what would actually
get faster if I spent money here?*

Method: scale each parameter by ±10% in `../C2/perf_model.py`, re-evaluate T, take the ratio.
Parameters that are physically linked (the two DRAM links; the four array ports) are
also scaled as groups, because widening one alone is meaningless.

## The result

| Parameter | **prefill kernel** | **decode kernel** | (whole prefill) | (whole decode step) |
|---|---|---|---|---|
| DRAM ↔ DMA ↔ SRAM path (both links) | 0.00 | **−1.01** | 0.00 | **−1.01** |
| one DRAM link alone | 0.00 | −0.55 | 0.00 | −0.55 |
| SRAM ↔ array bandwidth (all 4 ports) | **−1.01** | 0.00 | **−1.00** | 0.00 |
| ↳ the two 8 B/cycle **output** ports | **−0.64** | 0 | −0.37 | 0 |
| ↳ the two input ports | −0.46 | 0 | −0.70 | 0 |
| ↳ **32×16 output port alone** | **−0.34** | 0 | −0.35 | 0 |
| SRAM ↔ vector CPU | 0.00 | 0.00 | −0.01 | 0.00 |
| control-CPU latencies | 0.00 | 0.00 | 0.00 | 0.00 |
| DRAM read/write latency | 0.00 | 0.00 | 0.00 | 0.00 |

Two clean −1.0 entries, in different rows for the two phases. That is the answer.

## 2.A Prefill — the array ports, and specifically the output ports

Prefill attention time is exactly inversely proportional to array-port bandwidth (−1.0).
No surprise: it's compute-bound and the arrays *are* their ports (0.2b).

The interesting part is the split. The **output** ports carry more of it (−0.64) than
the inputs (−0.46), and the reason is structural:

- Q·Kᵀ contracts over only d = 128. A 32×16 tile therefore **loads in 64 cycles but
  drains in 128** through the 8 B/cycle output port. The array literally spends half its
  time waiting to write its own results.
- And the 16×16 array at K = 128 is exactly balanced, so its output port matters just as
  much as its input.

**The single most elastic scalar parameter is the 32×16 array's output bandwidth, at
−0.34.** If you could change one number in Figure 1, that is the one.

There is a nice irony here: the 32×16 array is the *bigger, more expensive* array, and
at the model's head dimension it delivers exactly the same 512 MAC/cycle as the small
one — because of a wire that is 8× narrower than the one feeding it.

Everything else has slack: vector CPU 38%, DRAM 5%.

**Note the zero on the control-CPU latencies.** That is not because they're inherently
harmless — it's *because commands are queued* (A7). Take queueing away and a 100-cycle
wait per 128-cycle tile drops peak from 1536 to **862 MAC/cycle**, and the layer from
226 ms to **476 ms**. The zero in that row is an achievement of the kernel design, not a
property of the hardware. Worth stating explicitly, because a table of zeros can look
like "these parameters don't matter" when the truth is "these parameters have been
neutralised."

## 2.B Decode — the DRAM → DMA → SRAM path

9 × 10⁹ MACs while moving 4.6 GB per step. It is a pure stream: **−1.0 on the DRAM
path.** The vector CPU is half idle and the arrays aren't involved at all.

Each link alone is **−0.55**, not −1.0, because **they are in series**. Widening one
without the other just moves the constriction. (The two −0.55s don't sum to exactly
−1.01 because of the small fill costs, but the message is clear: they must be widened
together. This directly shapes Improvement 1 in Problem 4.)

The right-hand columns show the same two parameters govern the *complete* model, not
just attention: prefill −1.0 on array ports, decode −1.0 on DRAM.

---

# Problem 3 — How fast is it really?

> *"Provide estimations of the TTFT and the interactivity for this workload and
> architecture. These numbers must be justified by a performance model which takes into
> account the architectural constants detailed above."*

- **TTFT** = the prefill time.
- **Interactivity** = 1 ÷ (decode step time).

Same machinery, but they reduce to completely different formulas — one counts MACs, the
other counts bytes.

## 3.1 The model (`../C2/perf_model.py`)

**Pipes.** Four of them: DMA path 64 B/cyc, the two arrays (rates from 0.2b), vector CPU
128 B/cyc. Time on a pipe = bytes ÷ bandwidth. One 800-cycle fill cost per phase covers
command and DRAM latencies (A7).

**Phases.** A layer = QKV projection → attention → O projection → gate+up → down →
element-wise (RoPE, residuals, SwiGLU). Within a phase the pipes run concurrently
because everything is double-buffered (A13), so:

> phase time = **max**(array time, DRAM time, vector time) + fill

Phases themselves are serialised.

**Matmuls in prefill — "weights stationary".** An 8 MiB weight slab sits in SRAM while
the 32 768-row activation matrix streams past it; the activations are re-read once per
slab (A9). Both arrays work on disjoint output tiles. The vector CPU requantises inputs
and accumulates each 256-deep chunk's int16 partial into fp32 (A4).
→ **time = MACs ÷ 1536 per cycle.** DRAM and vector both have slack.

**Matmuls in decode — "activations stationary".** Now the activation is a tiny [16 × K]
tile that stays resident, and the *weights* stream past once, split by output column
between the arrays.
→ **time = weight bytes ÷ 64 B per cycle.** The arrays end up 67% busy (16 MAC/B against
the 24 ridge — see Part 1).

Note the inversion: **which operand stays put flips between the two phases.** Whichever
one is reused is the one you keep.

**Attention:** the two kernels of Problem 1.

**Totals.** Prefill = 32 layers at M = 32 768 + the LM head on the 16 final tokens.
Decode step = 32 layers at M = 16 + LM head, evaluated at every context from 2049 to 2304.

## 3.2 Prefill → TTFT

One layer, M = 32 768 tokens (times in ms):

| Phase | time | array | DRAM | vector | bound by |
|---|---|---|---|---|---|
| QKV projection | 537 | 537 | 21 | 264 | arrays |
| flash-attention (Part A) | 226 | 226 | 10 | 87 | arrays |
| O projection | 358 | 358 | 15 | 177 | arrays |
| gate + up | 2 505 | 2 505 | 67 | 1 222 | arrays |
| down | 1 253 | 1 253 | 79 | 605 | arrays |
| element-wise | 36 | – | – | 36 | vector |
| **per layer** | **4 916** | | | | |

Look down the DRAM column: 21, 10, 15, 67, 79. Against array times 20–40× larger.
**Prefill barely uses memory bandwidth at all.**

> **TTFT ≈ 32 × 4.916 s + 8 ms = 157 s**

Sanity check in closed form — since it's compute-bound, just divide total work by peak:

```
2.29 × 10¹⁴ MACs ÷ 1.536 × 10¹² MAC/s = 149 s
```

157 vs 149 is a 5% gap, which is the kernel inefficiency: attention runs at 89% of peak,
plus chunk overheads. The two numbers agreeing is the model validating itself.

99% of TTFT is array time. Vector CPU ≈ 49% (mostly the fp32 accumulation that A4
forces), DRAM 3–6%.

**A free win worth noting.** Prefill is compute-bound and the 16 prompts don't depend on
each other. So prefilling them *one after another* instead of as one batch gives the
first user a TTFT of ≈ 10 s and the last ≈ 157 s — **at no cost in total time.** Same
work, same finish, dramatically better experience for 15 of the 16 users. Batching
helps memory-bound work; here it only delays everybody equally.

## 3.3 Decode → interactivity

One layer at context 2049 (times in ms):

| Phase | time | array | DRAM | vector | bound by |
|---|---|---|---|---|---|
| QKV projection | 0.40 | 0.26 | 0.39 | 0.13 | DRAM |
| attention (Part B) | 2.10 | – | 2.10 | 1.05 | DRAM |
| O projection | 0.26 | 0.17 | 0.26 | 0.09 | DRAM |
| gate + up | 1.84 | 1.22 | 1.84 | 0.60 | DRAM |
| down | 0.92 | 0.61 | 0.92 | 0.30 | DRAM |
| element-wise | 0.02 | – | – | 0.02 | vector |
| **per layer** | **5.53** | | | | |

Every row says DRAM. And "time" equals the DRAM column in every row — the definition of
memory-bound.

So the whole step collapses to a byte count:

> **T_step(L) ≈ (7.50 GB weights + 16 × L × 128 KiB of KV) ÷ 64 GB/s**
> **      = 117 ms + 0.033 ms × L**

| context L | step time | |
|---|---|---|
| 2049 | 185 ms | first generated token |
| **2176** | **189 ms** | average |
| 2304 | 194 ms | last token |

> **Interactivity ≈ 5.3 tokens/s per sequence, 84.5 tokens/s aggregate.**

62% of each step is streaming weights, 38% is streaming KV cache. The arrays are 67%
busy during the matmuls and completely idle during attention.

End to end: 157 s + 256 × 0.189 s ≈ **206 s** for the whole request.

Which is worth pausing on: **prefill is 76% of the total wall time** and produces exactly
one token. The 256 tokens of actual output take less time than reading the prompt.

## 3.4 How much do the assumptions matter?

An honest model states what would change if its assumptions were wrong:

| Assumption changed | TTFT | interactivity |
|---|---|---|
| **A2**: arrays PE-bound (1 MAC/PE/cycle) rather than port-bound | 315 s (**×2**) | unchanged |
| **A3**: exact 16-bit activations via hi/lo-byte split | 328 s (**×2.1**) | unchanged |
| **A7**: tile commands not queued | ≈ ×1.6 | 211 ms/token (−10%) |

The pattern is consistent: **array assumptions move TTFT by up to 2× and interactivity
not at all**, because decode never comes near the array limit. And crucially, none of
them change *which* resource is the bottleneck. So the answers to Problems 2, 4 and 5
survive even if these numbers move.

---

# Problem 4 — Two ways to make it better

> *"Suggest 2 architectural improvements to the accelerator that would improve the
> interactivity."*

**Read the question carefully: *interactivity*, not TTFT.** Interactivity is a decode
metric. So any improvement aimed at the arrays — the thing that dominates prefill — is
off-target here, however tempting.

3.3 gave the entire decode step in one line:

> T_step = (weight bytes + KV bytes) ÷ DRAM-path bandwidth

Exactly two levers exist: make the **denominator** bigger, or the **numerator** smaller.
Both are taken below.

| Variant | step | tok/s per seq | aggregate | then bound by | TTFT |
|---|---|---|---|---|---|
| baseline | 189 ms | 5.3 | 84.5 | DRAM path | 157 s |
| **1.** memory path ×4 (256 GB/s) | 115 ms | **8.7** | 140 | array input ports (weights), vector port (KV) | 157 s |
| 1′. same, ×8 | 115 ms | 8.7 | 140 | *unchanged* — DRAM is no longer the limit | 157 s |
| **2.** int4 weights + int8 KV, de-quantised in the DMA | 115 ms | **8.7** | 140 | array input ports | 157 s |
| 1 + 2 | 97 ms | 10.3 | 165 | array input ports | 157 s |
| 1 + 2 + array & vector ports ×2 | 49 ms | **20.6** | 330 | arrays | ≈ 79 s |

Note rows 2 and 3: **×4 and ×8 give identical results.** That is the model telling you
where the next wall is, and it's the most useful thing in the table.

## Improvement 1 — widen the memory path

The parameter with elasticity −1 in decode (2.B). Replace the two 512-bit interfaces
with an HBM-class path.

**189 ms → 115 ms, i.e. +65% interactivity.**

Two things to get right:

- **Both links must be widened together.** They're in series; the −0.55 single-link
  elasticities say widening one alone is half-wasted.
- **Stop at about 1.5×.** Past that, two *other* limits appear: the arrays can only
  ingest weights at 96 B/cycle (0.2c), and the vector CPU ingests KV at 128 B/cycle.
  Those become binding, which is precisely why ×4 and ×8 are the same number. Buying
  ×8 memory would be spending money on a resource that is already not the constraint.

**Prefill is completely untouched** — it never used the DRAM path.

## Improvement 2 — move fewer bytes: de-quantise int4 weights and int8 KV in the DMA engine

Attack the numerator instead. Store weights as int4 and the KV cache as int8 in DRAM,
and have the **DMA engine expand them on the way into SRAM**.

- weight traffic: 7.5 → **3.75 GB** per step
- KV traffic: 4.6 → **2.3 GB** per step

The elegance is where the expansion happens. The arrays and the vector CPU still see
int8 and int16 in SRAM exactly as before, so **the Problem 1 kernels need no changes at
all.** The narrow format exists only on the wire that was the bottleneck.

Alone: 189 → **115 ms**. Combined with Improvement 1: **97 ms, +95% interactivity.**

Honest caveats:
- int4 weights are standard practice for Llama-class models — low risk.
- **KV compression is a genuine accuracy trade-off** against the 16-bit activation spec,
  and must be validated on the real model before shipping. Stating that is part of the
  answer; a proposal that hides its risk isn't an engineering proposal.

## Things considered and not chosen

Worth listing, because rejections carry information:

- **32-bit accumulation inside the arrays across K-chunks.** Would remove 47% of vector
  traffic in the matmuls and the int16 partial-sum rounding. Good change — but it helps
  prefill, not interactivity.
- **Wider array ports, or a third array.** This is the *only* class of change that
  improves TTFT (last row of the table: TTFT 157 → 79 s). Also the wrong answer to *this*
  question.
- **Direct accelerator-to-accelerator links** — see Problem 5.
- **A bigger SRAM. Does not help decode at all.** Neither 7.5 GB of weights nor 4.8 GB of
  KV can ever be made resident, and the decode kernels use under 1 MiB as it is. This is
  the kind of upgrade that sounds obviously good and does nothing — the model is what
  tells you so.

---

# Problem 5 — Four accelerators

> *"Suppose that 4 such accelerators can be connected to a single host CPU."*

Recall constraint 0.2(f): **they can only talk through the host.** No direct links. Every
exchange is up to the host and back down, and it lands on the critical path.

## 5.1 How to parallelise

**The two phases want opposite partitions.** So the plan is
**data-parallel prefill → re-shard → tensor-parallel decode.**

### Decode: tensor parallelism

Step time = bytes *per accelerator* per token (3.3). So to go faster, both the **weights
and the KV cache** must be split. Splitting the batch does not split the weights.

| Scheme | weights/acc | KV/acc | step | tok/s per seq | comms per step |
|---|---|---|---|---|---|
| Pipeline (8 layers each) | 1.9 GB | 1.1 GB | **189 ms** | 5.3 | 3 × 128 KiB |
| Data (4 sequences each) | 7.5 GB | 1.1 GB | 135 ms | 7.4 | none |
| **Tensor, TP = 4** | 1.9 GB | 1.1 GB | **48 ms** | **21.0** | 64 all-reduces × 128 KiB |

Why the losers lose:

- **Pipeline** splits the weights 4 ways, but the four stages run *in sequence* for any
  given token — it still crosses all 32 layers. Throughput ×4, **interactivity
  unchanged.** A classic trap: the aggregate number improves and the metric you were
  asked about doesn't.
- **Data parallel** splits the KV cache but every accelerator still holds and streams
  **all 7.5 GB of weights**. Since weights are 62% of the step, you only remove the
  other 38%.
- **Tensor parallel** splits both. 189 → 48 ms.

The sharding:
- 8 query heads and 2 KV heads each. **The GQA groups divide cleanly by 4**, so the
  Part B kernel runs unmodified on its own groups — no code changes.
- one quarter of each FFN matrix (column-split gate/up, row-split down)
- one quarter of the vocabulary

Cost: **two all-reduces per layer** — after the O projection and after the down
projection — of 16 × 4096 × 2 B = **128 KiB** each. Plus a tiny arg-max exchange for the
split LM head. 32 layers × 2 = **64 all-reduces per token.**

And because of 0.2(f), each all-reduce means: all four accelerators send 128 KiB up, the
host sums, the host sends 128 KiB back (A12).

### Prefill: data parallelism

Apply the same TP sharding to prefill and the compute drops to a lovely 39.6 s — but
now each all-reduce carries **256 MiB** per accelerator instead of 128 KiB, because there
are 32 768 tokens in flight rather than 16. At the link bandwidth derived in 5.3, the 64
of them cost **42 s** — more than the compute. TTFT would *double*.

So prefill goes **data parallel**: 4 prompts per accelerator, full model on each,
**39.3 s, zero communication.** The Part A kernel runs unchanged.

This is the crux of the whole problem: *the same sharding is optimal for one phase and
catastrophic for the other*, because activation volume scales with tokens-in-flight while
weight volume doesn't.

### The transition

After prefill, each accelerator holds the complete KV cache of its own 4 sequences.
Decode needs each to hold 2 KV heads of **all 16**. So each sends 3/4 of its ≈ 1 GiB
through the host, once: **≈ 1 s**.

One second, paid once, to switch strategy. Against a TTFT that drops **157 s → ≈ 40 s**
and a step time that drops 189 → 48 ms. Easy trade.

## 5.2 Theoretical maximum interactivity

> *"What is the theoretical maximum interactivity that can be achieved by this system?"*

"Theoretical maximum" = assume communication is free. Then the TP = 4 step is just each
accelerator's own byte count over its own 64 GB/s path:

```
1.9 GB weights + 1.1 GB KV  =  3.0 GB  ÷  64 GB/s  =  47.6 ms
```

Still DRAM-bound, and comfortably so — array ingest would allow 19 ms and vector KV
ingest 9 ms. Splitting the work 4 ways did not change *which* resource binds.

> **≈ 21 tokens/s per sequence (47.6 ms/token), 336 tokens/s aggregate.**

Exactly 4× a single accelerator: four DRAM interfaces used perfectly in parallel. Which
is the sanity check — for a purely memory-bound workload with perfect sharding and free
communication, linear scaling is the expected answer.

## 5.3 Minimum host bandwidth for 70% of that maximum

> *"What is the minimum required data bandwidth between the host CPU and the 4
> accelerators to achieve 70% of the theoretical maximum interactivity?"*

Work backwards from the target. Four steps:

**1. Convert 70% into a time budget.**
```
70% of 21.0 tok/s = 14.7 tok/s   →   step may take 47.6 / 0.7 = 68.0 ms
```

**2. Subtract the compute; what's left is the comms budget.**
```
68.0 − 47.6 = 20.4 ms  for all communication in one token
```

**3. Divide by how many exchanges there are.**
```
20.4 ms ÷ 64 all-reduces = 319 µs per all-reduce
```

**4. Turn that into bandwidth.** In those 319 µs, each accelerator's link must carry
128 KiB up **and** 128 KiB back — *sequentially*, because the host cannot return a sum
before all four partials have arrived:

```
B ≥ 2 × 128 KiB ÷ 319 µs
```

> **≈ 0.82 GB/s per accelerator (≈ 6.6 Gbit/s), ≈ 3.3 GB/s aggregate at the host.**

Which is roughly **PCIe 3.0 ×4 per accelerator** — an entirely ordinary link. The
reassuring conclusion: this design does *not* need exotic interconnect. Because the
model is memory-bound, the activations that need exchanging are tiny (128 KiB) compared
to the weights being streamed (1.9 GB), so communication is a rounding error.

**Sensitivities** (the interesting part — the headline number alone isn't robust):

| Condition | requirement |
|---|---|
| 70% of maximum (the answer) | 0.82 GB/s |
| **90%** of maximum | **3.2 GB/s** — 4× the bandwidth for 20 more points |
| 5 µs per-transfer latency | 0.85 GB/s |
| 20 µs per-transfer latency | 0.94 GB/s |
| 32-bit instead of 16-bit partial sums | 1.65 GB/s (exactly double) |

The 90% row is the one to notice: the cost curve is steep near the ceiling, which is
normal for anything defined as a fraction of an asymptote.

All of these are **conservative**, because the model serialises each all-reduce with the
weight stream (A12). Prefetching the next weight slab (≤ 8 MiB, 0.13 ms) *during* each
all-reduce would recover up to ≈ 8 ms per step — the communication would hide almost
entirely behind work that has to happen anyway.

And prefill needs **no** host bandwidth at all beyond the one-off 1 GiB re-shard, since
it runs data-parallel with zero communication.

---

# Appendix — Assumptions, and why each one matters

Not just what was assumed, but what breaks if it's wrong.

| | Assumption | If it's wrong |
|---|---|---|
| **A1** | Single 1 GHz clock; links full-duplex at the stated rate; no bank conflicts (given) | — |
| **A2** | Array throughput is set by its **ports**, not its multipliers (0.2b) | If PE-bound instead (1 MAC/PE/cycle), all compute-bound numbers **double**: TTFT ≈ 315 s. Decode unaffected. |
| **A3** | Array operands dynamically requantised to int8; 16-bit kept for storage, residual stream, softmax stats, KV cache | The exact alternative (hi/lo-byte split) **doubles TTFT to ≈ 328 s**. Decode unchanged. |
| **A4** | Array outputs are int16 with a programmable shift; arrays **cannot** accumulate into an existing C tile, so 256-deep partial sums are accumulated in fp32 by the vector CPU | This is why the vector CPU is ~49% busy in prefill. In-array fp32 accumulation would remove 47% of that traffic. |
| **A5** | Tile commands take (base, row stride) for A, B, C | Needed for the strided writes in `issue_S_subtile_*`. |
| **A6** | DMA descriptors are queued, 2-D strided; DRAM latency paid once per descriptor | Without it, decode's streaming design collapses. |
| **A7** | Control CPU issues commands **asynchronously** into FIFOs | Without queueing, peak falls 1536 → **862 MAC/cycle**. This assumption is the reason the control latencies show elasticity 0 in Problem 2. |
| **A8** | Vector CPU: infinite compute (given), fp32, cost = bytes ÷ 128. RMSNorm fused into the requantisation pass | Fusing is what keeps the vector CPU off the critical path. |
| **A9** | Prefill matmuls: 8 MiB weight slab resident, activations re-read per slab. Decode matmuls: activations resident, weights streamed once | This is the "which operand stays put" inversion of 3.1. |
| **A10** | Decode attention runs on the **vector CPU**, not the arrays | Justified at length in 1.B. |
| **A11** | KV cache 16-bit, **token-major** in DRAM. Weights int8, stored transposed | Token-major is chosen for decode's single contiguous append; prefill pays to transpose V. |
| **A12** | Accelerators communicate only through the host; all-reduces are host-mediated and on the critical path | Makes 5.3's numbers conservative. |
| **A13** | Within a phase, DMA/arrays/vector overlap perfectly; phases are serialised | Two approximations in **opposite** directions — see Limitations. |
| **A14** | Greedy sampling; embedding lookup is a DMA gather | Both negligible. |
| **A15** | Llama 3.1 8B: 32 layers, d = 4096, 32 query / 8 KV heads of 128, FFN 14 336 (SwiGLU), vocab 128 256, untied LM head → 8.03 B params | — |

## Limitations, stated plainly

**The model is analytical, not cycle-accurate.** Its two biggest approximations point in
*opposite* directions — perfect overlap within a phase is optimistic, zero overlap
between phases is pessimistic — so they partly cancel, but neither is verified. The next
step would be a discrete-event simulation of the pseudocode's command streams to check
the 6144-cycle prefill block schedule, the decode ring depth, and the buffer sizes.

**The int8 quantisation error at the array boundary (A3) has not been evaluated
numerically.** It needs a calibration run on the real model. All the timing numbers
assume it's acceptable; that is an assumption, not a result.

**The int16 partial-sum output (A4) is a precision risk** for the 256-deep FFN sums.
fp32 accumulation on the vector CPU bounds it to per-chunk rounding, but does not
eliminate it.

---

# The whole thing in one page

1. **One ratio defines the machine: 24 MAC per DRAM byte.** Everything is "which side
   of 24 is this?"
2. **Prefill sits far above it** (32 768 MAC/byte on matmuls, 460 on attention) →
   compute-bound → the arrays are the resource → **157 s TTFT**.
3. **Decode sits far below it** (16 and 2) → memory-bound → the DRAM pipe is the
   resource → **189 ms/token, 5.3 tok/s**.
4. **So there are two kernels, not one.** Prefill tiles and software-pipelines to keep
   both arrays at 89% of peak; decode abandons the arrays entirely and streams the KV
   cache through the vector CPU.
5. **The bottlenecks, measured by elasticity:** array output ports for prefill (−1.0,
   with the 32×16's output port the single worst scalar at −0.34); the DRAM path for
   decode (−1.0, and −0.55 for either link alone because they're in series).
6. **To improve interactivity**, attack decode's byte count or its bandwidth: widen the
   memory path (+65%) and de-quantise int4/int8 in the DMA engine (together +95%). Not
   the arrays — they're already idle. Not more SRAM — nothing would fit anyway.
7. **Across 4 accelerators**, the phases want opposite partitions: data-parallel prefill
   (zero comms, TTFT 157 → 40 s), then a 1-second re-shard, then tensor-parallel decode
   (48 ms/token, 21 tok/s). Ceiling is 4× linear at 47.6 ms, and reaching 70% of it needs
   only ≈ 0.82 GB/s per accelerator — ordinary PCIe.

The recurring lesson: **find the one resource that actually binds, then arrange
everything else — layout, buffering, which engine runs the math, how the work is
sharded — so that resource never stops.** And re-check which resource that is whenever
the workload changes shape, because in this workload it changes twice.
