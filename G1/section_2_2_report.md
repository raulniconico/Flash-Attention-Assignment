# Section 2.2 — Flash-attention design and performance analysis

**Scope:** all five requirements in section 2.2 of the three supplied pages.  
**Status:** a documented design, tested numerical reference, and reproducible analytical estimate; not an implementation measured on the accelerator.  
**Privacy:** the attached pages were used only for this task. They were not published, sent to another person, or included in public searches. Public lookups used public model/algorithm names. The submission bundle contains new work, not copies of the source images.

## 1. Main findings and how to interpret them

The specification permits a useful but unusual solution: implement attention on the **vector CPU**, including its dot products, while retaining 16-bit activations. This follows from the explicit assumption that this generic SIMD processor has infinite computation capacity and is only memory-bound. The statement does not restrict it to softmax or prohibit matrix multiplication. This choice also avoids silently converting the required 16-bit activations into 8-bit array inputs.

The primary design below is optimized around that literal assumption. It is a concrete candidate, not a proof of global optimality: the vector register capacity, instruction set, and reduction implementation are not supplied. I assume at least 12 KiB of vector-local working storage, explain the access pattern, and charge all modeled SRAM traffic. If the assignment intended the vector CPU to perform only elementwise operations, the intended machine is different; the optional array mapping in section 5 explains what additional numerical and timing specifications are needed.

| Requested result | Conditional result from the supplied model |
|---|---:|
| Warm time to first token, batch 16, prompt 2048 | **167.74 s**, approximately **168 s** |
| Average subsequent-token latency | **204.84 ms** |
| Average interactivity per sequence | **4.88 tokens/s** |
| Aggregate throughput across the 16 sequences | **78.11 tokens/s** |
| Four-way tensor parallelism, modeled schedule, zero host-network time | **18.11 tokens/s per sequence** |
| Four-way ideal streaming-traffic ceiling | **21.21 tokens/s per sequence** |
| Host bandwidth for 70% of that traffic ceiling, serialized host collectives, zero fixed collective latency | **11.08 GB/s aggregate across both directions and all four links**, with FP32 reductions |

These are **conditional estimates**, not uniquely determined answers to an otherwise complete hardware specification. In particular, the four-device traffic ceiling assumes dense, uncompressed, exhaustive execution with streamed weights/KV and no persistent cross-step operand cache. The host-bandwidth result assumes a particular host gather/reduce/broadcast schedule. Other schedules or numerical formats produce different answers.

The precision of `results.json` is for reproducibility. The rounded numbers above are more appropriate for engineering communication.

## 2. Review of the complete supplied material

### 2.1 Page-by-page interpretation

| Source | What it establishes | Consequence for the design |
|---|---|---|
| `../../../Téléchargements/1.png` | Host interface, control CPU, DRAM, DMA, software-managed SRAM, two array shapes, vector CPU | Control orchestrates; compute units cannot dereference DRAM; all operands must be staged through DMA |
| `../../../Téléchargements/2.png` | Capacities, bandwidths, configuration/communication latencies, clock, overlap assumptions | The model must include transfers, startup, command granularity, SRAM capacity, and pipeline overlap |
| `../../../Téléchargements/3.png` | Llama 3.1 8B, W8/A16, batch 16, 2048-token prompt, 256 generated tokens; five questions; submission expectations | Account for the whole model when estimating TTFT, distinguish prefill/decode, provide code and documented assumptions |

There is a notation issue on page 1: two row-oriented operand buffers of shapes `R×K` and `16×K` cannot literally be multiplied in that order without transposing the second logical operand. I interpret the interface as producing all row-pair dot products, equivalent to `L @ R.T`. This interpretation matches the stated `R×16` output. The pseudocode explicitly packs the right operand for `PV`.

### 2.2 Architectural constants, in consistent units

At 1 GHz, one cycle is 1 ns. Bandwidth uses decimal GB/s; capacity uses binary GiB/MiB.

| Parameter | Given value | Value used |
|---|---:|---:|
| DRAM capacity | 64 GiB | 68,719,476,736 bytes |
| Scratchpad capacity | 16 MiB | 16,777,216 bytes |
| DRAM–DMA link | 512 bits/cycle | 64 bytes/cycle = 64 GB/s |
| DMA–SRAM link | 512 bits/cycle | 64 bytes/cycle = 64 GB/s |
| SRAM–vector link | 1024 bits/cycle | 128 bytes/cycle = 128 GB/s |
| SRAM–16×16 array input | 512 bits/cycle | 64 bytes/cycle |
| SRAM–32×16 array input | 768 bits/cycle | 96 bytes/cycle |
| Either array output link | 64 bits/cycle | 8 bytes/cycle |
| DMA configuration | 200 cycles | Charged per modeled DMA descriptor |
| DRAM first read byte | 200 cycles | Charged per modeled read |
| DRAM first write byte | 150 cycles | Charged per modeled write |
| Vector communication | 300 cycles | Charged per vector-program launch |
| Array communication | 100 cycles | Relevant to optional array backend |
| Array input/output widths | 8 / 16 bits | Not a native W8/A16 interface |

I conservatively interpret “bidirectional” as an aggregate bandwidth budget shared by reads and writes. The two 64-byte/cycle links are **in series**: end-to-end throughput is their minimum, not their sum. DMA and vector execution may overlap on different buffers. SRAM bank conflicts and contention are ignored as instructed. If the links instead provide 64 bytes/cycle independently in each direction, the model should be revised; read-dominated decode would change less than a balanced read/write workload.

### 2.3 Explicit assumptions and unresolved contracts

| Decision or missing item | Primary assumption | Why it matters |
|---|---|---|
| Model dimensions | 32 layers; hidden 4096; MLP 14336; 32 Q heads; 8 KV heads; head dimension 128; vocabulary 128256 | The three pages name the model but do not enumerate these dimensions. They are explicit model inputs to confirm against the deployed checkpoint |
| 16-bit activation encoding | A vector-supported 16-bit real-valued representation; FP32 intermediate sums/state | Bit width alone does not identify FP16, BF16, or scaled integer semantics |
| W8 format | Symmetric signed 8-bit weights with one FP32 scale per output channel | Scale reads are charged by the model; different quantizers may require more metadata and work |
| Vector storage | At least 12 KiB local working storage beyond SRAM | Allows an 8 KiB accumulator plus streamed operands; a smaller register file causes spills and lowers throughput |
| Vector arithmetic | Exactly the supplied memory-only timing assumption | No finite SIMD FLOP-rate term is invented |
| DMA | Large contiguous transfers; double buffering; serialized aggregate link traffic | Avoids assuming that compute can read DRAM directly |
| Array math/timing | Unspecified; no array timing used in the primary estimate | Array dimensions and port widths alone do not establish MACs/cycle or pipeline initiation intervals |
| Weight residency | Model preloaded in accelerator DRAM | Host loading time cannot be estimated without host bandwidth |
| Sampling | Greedy decoding on the accelerator | Gives a specific, cheap final vocabulary selection and a small four-way reduction |
| TTFT | Prompt available on the device; first generated token follows prefill logits | Host tokenization/queueing/transfers are separate unknown terms |
| Decode count | 255 single-token forwards after prefill | Prefill already produces generated token 1 of 256 |
| Four-device topology | Communication relayed through host, no direct peer link | Defines the collective traffic calculation |
| FP32 collectives | Row-parallel partial results reduced in FP32 | Avoids assuming that summing quantized 16-bit partials preserves accuracy |

The official model page is [Meta’s Llama 3.1 8B Instruct page](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct). The detailed configuration endpoint could not be retrieved during this review; the dimensions above are stated inputs, not a claim of checkpoint inspection. No weights were downloaded.

## 3. Workload sizes and arithmetic

Let `B=16`, `S=2048`, `L=32`, `D=4096`, `F=14336`, `Hq=32`, `Hkv=8`, and `d=128`. Four query heads share each KV head. Count one multiply-accumulate as one MAC; a conventional FLOP count is twice the MAC count.

### 3.1 Linear layers

| Operation | Weight shape, input × output | Parameters |
|---|---:|---:|
| Q projection | 4096 × 4096 | 16,777,216 |
| K projection | 4096 × 1024 | 4,194,304 |
| V projection | 4096 × 1024 | 4,194,304 |
| Attention output | 4096 × 4096 | 16,777,216 |
| MLP gate | 4096 × 14336 | 58,720,256 |
| MLP up | 4096 × 14336 | 58,720,256 |
| MLP down | 14336 × 4096 | 58,720,256 |
| **Per transformer layer** | | **218,103,808** |

Thus the transformer linears contain `32 × 218,103,808 = 6,979,321,856` weights. The vocabulary projection contains `4096 × 128256 = 525,336,576` weights. The repeatedly streamed decode weights total **7,504,658,432 bytes**, before scale metadata. Including the separate input embedding gives **8,029,995,008 bytes** of large weight tensors. Small normalization parameters do not materially change capacity or the timing budget.

The prefill transformer linears require

\[
M_{\mathrm{dense,prefill}}=BSL(218{,}103{,}808)
=228{,}698{,}418{,}577{,}408\ \mathrm{MAC}.
\]

Only the last prompt position of each sequence needs the vocabulary projection to select the first generated token. Applying the vocabulary projection to all 2048 prompt positions would unnecessarily inflate TTFT.

Every decode forward requires `B × 7,504,658,432 = 120,074,534,912` dense MACs.

### 3.2 Attention and KV cache

Causal prefill has `S(S+1)/2` valid query/key pairs per head. Including both `QKᵀ` and `PV`,

\[
M_{\mathrm{attention,prefill}}=LBH_qdS(S+1)
=8{,}800{,}387{,}989{,}504\ \mathrm{MAC}.
\]

At decode context `C`,

\[
M_{\mathrm{attention,decode}}=2LBH_qdC.
\]

At `C=2176`, this is **9,126,805,504 MACs**. The cache contains KV heads, not query heads:

\[
D_{\mathrm{KV}}(C)=2LBH_{kv}dC(2\ \mathrm{bytes}).
\]

| Context | Total 16-bit KV storage |
|---|---:|
| 2048 | 4,294,967,296 bytes = 4 GiB |
| 2176 | 4,563,402,752 bytes = 4.25 GiB |
| 2304, capacity reservation | 4,831,838,208 bytes = 4.5 GiB |

Weights, KV, and working tensors fit comfortably within 64 GiB. The full model does not fit within 16 MiB SRAM. The largest simple prefill activation, `32768×14336×2`, is about 0.875 GiB, so several DRAM working buffers remain practical. At the end of generation the final emitted token need not itself be forwarded; reserving cache capacity through 2304 is convenient even though the modeled decode forwards use contexts **2049 through 2303**.

## 4. Requirement 1 — Kernel, SRAM, tiling, and dataflow

### 4.1 Algorithm and numerical invariant

The algorithm uses tiled attention with a stable online softmax, retaining the running output numerator rather than storing the full attention matrix. The method and unnormalized-output formulation are described in the authors’ [FlashAttention paper](https://arxiv.org/abs/2205.14135) and [FlashAttention-2 paper](https://arxiv.org/abs/2307.08691). The hardware layout, schedules, traffic accounting, and backend decisions below are specific to this assignment.

For each query row maintain maximum `m`, exponential sum `ell`, and output numerator `U`. Initialize `m=-∞`, `ell=0`, `U=0`. For a block of scaled, masked logits `s`:

\[
\begin{aligned}
m'&=\max(m,\max_j s_j),\\
\alpha&=\begin{cases}0&\ell=0\\e^{m-m'}&\ell>0\end{cases},\\
p_j&=e^{s_j-m'},\\
\ell'&=\alpha\ell+\sum_jp_j,\\
U'&=\alpha U+\sum_jp_jV_j.
\end{aligned}
\]

After the final block, output `O=U/ell`, rounded to the specified 16-bit activation format. The invariant is that `ell` and `U` contain the denominator and numerator for all visited keys using the same current maximum. Rescaling both preserves the normalized result. A wholly masked block leaves the state unchanged; this prevents undefined `-∞ - -∞`. The numerical reference handles the empty-row convention explicitly, although ordinary causal self-attention includes at least the current token.

This preserves the mathematical attention operation. It does not promise bitwise equivalence to a different reduction order, and the supplied test does not validate a particular quantizer or FP16/BF16 rounding implementation.

### 4.2 SRAM allocation

For one sequence and one KV head, all 2048 keys and values occupy just 1 MiB. Its four associated query heads occupy 2 MiB; the corresponding 16-bit outputs occupy another 2 MiB. This is small enough to keep an entire head group on chip.

| Allocation | Size | Purpose |
|---|---:|---|
| Bundle 0: K/V | 1 MiB | Reuse across all four query heads |
| Bundle 0: Q | 2 MiB | All prompt queries for the head group |
| Bundle 0: O | 2 MiB | Completed 16-bit output rows |
| Bundle 1: K/V, Q, O | 5 MiB | DMA/compute overlap with the next head group |
| FP32 tile workspace | 0.5 MiB | Scores, probabilities, numerator, softmax statistics |
| Descriptors, alignment, miscellaneous | 0.5 MiB | Control-managed workspace |
| Unallocated reserve | 5 MiB | Additional buffers or alternative tilings |
| **Total** | **16 MiB** | Capacity respected |

Only two 5 MiB bundles and 1 MiB workspace are committed. The numerical microtile needs considerably less than the reserved 0.5 MiB. Reserving the extra space avoids relying on packed, unaligned allocations. Vector-local storage is a separate, explicit hardware assumption; the given 16 MiB SRAM does not establish that register capacity.

For decode, only four current query vectors and outputs are needed per KV group. Even at the end of generation, its K/V occupy about 1.125 MiB. The decode allocation therefore also fits easily.

### 4.3 Tile and loop choices

Use **16 query rows × 128 keys** for prefill. The feature reduction dimension is 128.

1. **Select batch item and KV head as the outer job.** Load K/V once and reuse them for four query heads; avoid multiplying KV traffic by the GQA ratio.
2. **Keep the whole head group resident.** A schedule that reloads K/V from DRAM for every 16-row query tile wastes the available SRAM.
3. **Keep one query block’s FP32 numerator resident in SRAM.** Sweep key blocks in order and normalize once at the end.
4. **Skip wholly future key blocks.** For diagonal blocks, mask invalid elements before exponentiation. Dense microtiles still compute some masked entries; the timing model charges the full visited tile.
5. **Keep the 16×128 matrix accumulator local during each dot-product reduction.** Its FP32 footprint is 8192 bytes. The assumed 12 KiB local budget includes additional streamed operands and temporaries.
6. **Use a distinct decode microkernel.** It processes the four query heads sharing K/V together. Their Q, numerator, statistics, and a 4×128 score/probability block fit within the local budget. Each K/V element is read once from SRAM for the group.

The primary vector kernel is not constrained by the arrays’ `K≤256` rule. The optional array kernel chooses a 128-key tile so its `PV` reduction is also legal.

### 4.4 Control-CPU responsibilities

The control CPU allocates buffers, submits DMA descriptors, launches vector programs, and waits for completion events. It does **not** execute the attention arithmetic. A vector program contains the inner tile loops, so there is one 300-cycle launch per resident head-group job, rather than one launch per scalar operation or every tile.

The two bundles follow `FREE → LOADING → READY → COMPUTING → STORING → FREE`. There is only one vector CPU and one modeled DMA path. Input transfer into the free bundle can overlap computation on the other bundle. A buffer cannot be reused until its output write completes. The 0.5 MiB tile workspace is shared only because vector programs execute serially.

The complete control pseudocode, vector programs, layouts, and optional array backend are in **`kernel_pseudocode.txt`**. This distinction between host, control, and vector work is intentional: the required implementation runs as an orchestrator on the control CPU while dispatching its arithmetic to the compute units.

### 4.5 Accuracy validation

`reference_attention.py` independently compares blocked online attention against ordinary full-row softmax attention. It covers 19 combinations including causal/noncausal execution, multiple GQA heads, uneven block tails, blocks larger than the sequence, one-token decoding, all-masked rows, and very large logits. Additional checks validate combining split-KV states and empty states.

Observed maximum absolute difference: **1.11×10⁻¹⁶** using Python floating-point arithmetic. This supports the recurrence and indexing. It is not evidence for device timing, quantization quality, or hardware implementation correctness.

## 5. Array mapping and the precision problem

The arrays remain useful hardware, but the stated specification is insufficient to make them the unconditional baseline.

### 5.1 Mapping if an 8-bit compute path is permitted

For `QKᵀ`, use `R=32` or `16` query rows and 16 key rows per array launch, reducing over head dimension 128. Eight column subtiles cover a 128-key block. For `PV`, use the probability block as `R×128`; pack each 16 output channels of V as `16×128` rows, then launch eight output-column subtiles. Vector kernels apply scales, masking, exponentials, and FP32 online accumulation.

Assign independent query-row jobs to either array based on completion time. Fixing one array permanently to QK and the other to PV needlessly ties load balance to one stage. Maintain separate input/output slots so the next compute can overlap the previous output drain as permitted by the assignment.

At decode, each sequence/KV group has only four query rows. Padding to 16 or 32 rows wastes most arithmetic. Queries from unrelated KV groups cannot simply be stacked into one ordinary GEMM sharing the same right operand. A grouped/batched hardware mode could help, but none is specified.

### 5.2 The 8-bit/16-bit mismatch is substantive

Q, K, V, and stored layer activations are 16-bit. Converting them to 8-bit changes the numerical computation. Quantizing the softmax probability block adds another approximation. The specification gives no error tolerance authorizing these changes.

Moreover, an arbitrary 8-bit dot product over 128 or 256 terms can exceed a signed 16-bit output range. For example, `256×128×128=4,194,304`, far above 32767. A 16-bit output might mean a scaled floating-point value after wide accumulation, saturation, truncation, or raw integer bits. These are different contracts. A robust implementation must not assume one silently.

Valid options are:

- Use the vector path to preserve the 16-bit operand representation, as in the primary design.
- Specify and validate an explicitly approximate 8-bit attention path, including scales and accumulator behavior.
- For a suitable integer representation, decompose operands into multiple bit/limb products and combine in wider vector arithmetic. This is not a free, one-pass native A16 operation. Floating-point encodings require different treatment.
- Split reductions into provably safe small pieces, then accumulate in the vector CPU; charge the extra launches and output traffic. For unrestricted signed INT8 operands and raw INT16 output, a one-term slice is always range-safe. This illustrates how costly a literal narrow-output contract can become.

### 5.3 What the ports establish—and what they do not

For a tile with reduction length `k`:

| Tile | Input bytes | Minimum input-transfer cycles | Output bytes | Minimum output-transfer cycles |
|---|---:|---:|---:|---:|
| 16×16 | 32k | k/2 | 512 | 64 |
| 32×16 | 48k | k/2 | 1024 | 128 |

These are port transfer bounds. Neither `k/2` nor `k` is established as the array compute time by the pages. Even a conventional one-MAC-per-PE-per-cycle assumption needs to be stated. The two-array 768-GMAC/s peak often inferred from the PE count is therefore conditional, not given.

For illustration only, assume one MAC per PE per cycle, input streaming hidden under compute, and 100 setup cycles added per tile while output draining overlaps the next operation. The modeled steady interval is `max(k+100, drain_cycles)`. At `k=128`, both arrays have a 228-cycle interval, and aggregate useful capacity with full tiles is about **431 GMAC/s**. The output link is then not the largest active bottleneck: increasing it does not shorten the 228-cycle interval. At `k=256`, corresponding full-tile capacity is about **552 GMAC/s**. If commands can overlap completely, the conclusion changes again.

These illustrative rates exclude precision-conversion costs and remaining dependencies. They must not be substituted for measured kernel throughput or mixed into the primary vector result.

## 6. Requirement 2 — Bottleneck and elasticity

Define the beneficial elasticity of a capacity parameter `x` as

\[
E_x=-\frac{\partial\ln T}{\partial\ln x}.
\]

An elasticity near 1 means a 1% increase in that capacity gives about a 1% runtime reduction locally. For latency parameters, this sign convention gives a negative value: increasing a latency makes the kernel slower. The script evaluates 1% finite differences.

| Parameter increased by 1% | Prefill attention elasticity | Decode attention elasticity | Entire decode step elasticity |
|---|---:|---:|---:|
| SRAM–vector bandwidth | 0.9998 | 0.0036 | 0.0128 |
| Both DRAM–DMA and DMA–SRAM bandwidths together | 0.00013 | 0.9307 | 0.9357 |
| DRAM–DMA bandwidth alone | 0 | 0 | 0 |
| DMA–SRAM bandwidth alone | 0 | 0 | 0 |

**Prefill attention is limited by the SRAM–vector interface in the primary implementation.** Resident K/V avoid repeated DRAM loading, but each query block still causes substantial SRAM traffic for dot products and FP32 state updates.

**Decode attention is limited by the end-to-end DRAM-to-SRAM path.** Four-head GQA reuse makes SRAM/vector traffic lower than the available vector bandwidth, while every current context requires reading the cache again.

The zero one-sided elasticities for either 64-byte/cycle link alone are not a contradiction. At equal series-link capacities, `min(B1,B2)` has a kink: increasing only one leaves the other bottleneck unchanged. There is a **joint bottleneck**, not a unique scalar hardware parameter with a smooth derivative. Reducing either link would hurt; improving both helps. If clock frequency is admitted as a parameter, fixed cycles-per-operation imply elasticity 1 for frequency globally, but increasing clock while keeping all per-cycle timings unchanged is itself a strong hardware assumption.

The assignment’s request for one universally largest-elasticity parameter has no unique answer across prefill, decode, and alternative numerical backends. Reporting those distinctions is necessary for a defensible answer.

## 7. Requirement 3 — Reproducible performance model

### 7.1 Primitive costs

Let `b=min(Bdram,Bdma_sram)` in bytes/cycle. The primary model uses

\[
\begin{aligned}
t_r(n)&=200+200+\lceil n/b\rceil,\\
t_w(n)&=200+150+\lceil n/b\rceil,\\
t_v(n)&=300+\lceil n/B_{vector}\rceil.
\end{aligned}
\]

The vector term includes memory accesses, not an invented finite arithmetic throughput. First-byte and descriptor costs are conservatively paid per descriptor. A real DMA engine may hide some latency using outstanding requests, which motivates improvement 2.

Independent pipeline stages use a maximum in steady state; dependent whole-model operations are summed. Using a single global `max(total_compute,total_memory)` would incorrectly overlap operations separated by attention, MLP, and layer dependencies.

### 7.2 Dense layer schedule and costs

A 256-row activation macrotile stays in SRAM while full-K, 128-column weight panels are double-buffered. The vector CPU processes 16×128 output microtiles with a local FP32 accumulator, then writes 16-bit results. For tensor-parallel row-partitioned outputs, it writes FP32 partials.

The worst primary dense allocation is below 16 MiB. For the down projection: activation `256×14336×2 = 7 MiB`; two weight panels `2×14336×128 = 3.5 MiB`; full output `256×4096×2 = 2 MiB`; reserve 1 MiB; total **13.5 MiB**. The program asserts the allocation for every modeled matrix.

For one macro with `r` rows, reduction `K`, and `N/128` output panels, a panel has

\[
\begin{aligned}
D_W&=128K+512,\\
D_V&=(r/16)128K+2rK+2r128+512.
\end{aligned}
\]

The 512 bytes are FP32 output-channel scales. The vector reads each weight panel once per 16 output rows and the activation values once per output panel. Set `R=t_r(D_W)` and `V=t_v(D_V)`. The macro cost is

\[
t_r(2rK)+R+V+(N/128-1)\max(R,V)+t_w(2rN).
\]

Input/output macrotile transfers are serialized with that macro’s panel pipeline. Weight-panel DMA overlaps vector work on the preceding panel. This explicit schedule avoids an unrealistically perfect global overlap assumption.

The leading vector traffic is approximately `(1/16 + 2/128)×MAC = (5/64)×MAC` bytes, or **12.8 MAC per SRAM byte**. With the assumed memory-only vector CPU, this schedule can exceed a conventional 768-GMAC/s aggregate array estimate. It does not imply such throughput for a real finite-throughput SIMD core.

### 7.3 Prefill attention traffic

For each visited 16-query × 128-key tile, the model charges:

| Operation | SRAM/vector bytes |
|---|---:|
| QK: read Q16 and K16; write S32 | 45,056 |
| Online update: read S32; write P32; read/write U32 and m/l | 33,024 |
| PV: read P32 and V16; read/write U32 | 57,344 |
| **Per visited tile** | **135,424** |

The counter includes `1088` visited tiles per query head: eight 16-row query blocks per 128-key diagonal block, giving `8×(1+…+16)`. It counts padding/masked arithmetic in diagonal tiles. Initialization and final conversion are also charged.

One KV group contains four query heads. Its vector program costs **4,687,148 cycles**, including a single launch. Loading K/V and Q and writing O costs **83,070 cycles**. Thus group-level DMA is largely hidden under vector processing. The model includes a conservative first/last-job allowance for each layer rather than claiming a perfect steady state from the first cycle.

### 7.4 Decode attention traffic

One KV group reads `2×C×128×2` bytes of K/V, reads four query vectors, and writes four output vectors. A fused vector program keeps its four queries, numerator, maxima, and sums local, so K/V traverse SRAM–vector once each. It does not materialize the full score sequence in DRAM.

At mean context 2176, the whole model’s raw KV read lower bound is

\[
4{,}563{,}402{,}752/(64\times10^9)=71.303\ \mathrm{ms}.
\]

The scheduled estimate is **77.03 ms**, including per-group transfer/launch costs, small Q/O transfers, and boundaries. Large resident-group transfers amortize configuration much better than per-row DMA.

### 7.5 Full-model results

| Component | Warm prefill / first token | Mean decode forward |
|---|---:|---:|
| Transformer linears | 143.7458 s | Included below |
| Vocabulary head and greedy selection | 0.0088 s | Included in dense total |
| Dense total during decode | — | 125.3357 ms |
| Attention | 19.3512 s | 77.0282 ms |
| Auxiliary operations / transfers | 4.6349 s | 2.4754 ms |
| **Total** | **167.7407 s** | **204.8392 ms** |

The auxiliary budget includes two RMSNorms, two residual adds, rotary processing, SiLU/gating, embedding movement, and decode KV append. It conservatively budgets separate DMA and vector traffic for these passes; more aggressive fusion could reduce it. Layer operations execute in dependency order. Prefill QKV writes already populate the cache, so the model does not charge a second complete cache write.

Subsequent-token latency increases with context:

| Forward context | Latency | Per-sequence rate | Batch aggregate rate |
|---|---:|---:|---:|
| 2049 | 200.63 ms | 4.98 tokens/s | 79.75 tokens/s |
| 2176, average | 204.84 ms | 4.88 tokens/s | 78.11 tokens/s |
| 2303 | 209.05 ms | 4.78 tokens/s | 76.54 tokens/s |

The 255 decode forwards take approximately **52.23 s**. Including prefill, the 256-token response completes in about **219.97 s**, excluding host-side unknowns. The first-token latency is not `prefill + one ordinary decode step`: prefill already supplies the logits for that first token.

### 7.6 What these estimates exclude

The missing host terms should remain visible:

\[
TTFT_{observed}=T_{host\,preprocessing}+T_{input\,transfer}+T_{queue}+167.74\,s+T_{token\,return}.
\]

For a cold start, add at least the model-loading traffic divided by the effective host link bandwidth, plus loading/setup costs. No host bandwidth or latency is supplied, so that term cannot be made numerical. Alternative sampling, different activation encoding, finite vector arithmetic, register spills, weight quantization metadata beyond the chosen format, and software/layout costs outside the assumed packed layout also require model changes.

The code is a deterministic analytical calculator with explicit scheduling formulas. Calling it a cycle-accurate simulator or a measured profile would overstate the evidence.

## 8. Requirement 4 — Two architectural improvements

### Improvement 1: widen the complete DRAM-to-SRAM path

Increase **both** DRAM–DMA and DMA–SRAM from 512 to 1024 bits/cycle, providing 128 bytes/cycle end to end. A single widened link would leave the other at 64 bytes/cycle.

In the same model, mean decode latency falls from 204.84 to **122.24 ms** and interactivity rises from 4.88 to **8.18 tokens/s per sequence**, about **1.68×**. The speedup is below 2× because the vector stage becomes active for some dense panels and other overheads remain. This improvement preserves W8/A16 numerics.

### Improvement 2: autonomous DMA descriptors with multiple outstanding transfers

Add a hardware descriptor queue/block-list engine and enough outstanding requests to hide per-transfer first-byte latency. Allow the control CPU to submit a sequence of transfers and dependencies once while DMA streams later panels. This targets the repeated 200-cycle configuration and DRAM startup costs without changing the numerical workload.

An optimistic model that fully amortizes these recurring costs, while retaining transfer bytes and vector launches, gives **195.09 ms** and **5.13 tokens/s per sequence**, about **5%** better than baseline. This is an upper estimate of the isolated benefit: first/last transfers still have real latency and finite queues may not hide every gap. It is nevertheless directly tied to a measured-in-the-model source of overhead and is more modest in implementation scope than a multi-GiB on-chip cache.

Simply doubling SRAM–vector bandwidth barely improves decode here, to **4.91 tokens/s**, because the memory path remains dominant. It would substantially improve prefill. Wider/native A16 array arithmetic and wider array outputs are reasonable alternatives if the vector assumption is replaced by a conventional finite-throughput core; the present model does not justify prioritizing them for decoding.

## 9. Requirement 5 — Four accelerators

### 9.1 Select tensor parallelism for interactivity

Each accelerator has enough DRAM to hold the entire model, so data parallelism is feasible. Splitting the 16 sequences into four independent batches of four simplifies communication and improves service capacity, but each device still reads a full set of model weights per forward. It does not deliver four times the per-sequence interactivity in a weight-streaming regime.

Pipeline parallelism assigns groups of layers to devices, but a single sequence’s next token still traverses all stages in order. It principally improves utilization with multiple microbatches, rather than guaranteeing a fourfold latency reduction.

For this objective, use **four-way tensor parallelism**, retaining all 16 sequences on each rank while sharding tensors:

| Component | Per-accelerator assignment | Communication |
|---|---|---|
| Q projection / attention | 8 query heads | Local attention |
| K/V projection / cache | 2 KV heads | Four query heads still share each local KV head |
| Attention output projection | Input-channel/row partition | Sum 16×4096 FP32 partial output |
| MLP gate/up | 3584 intermediate channels | Local activation/gating |
| MLP down | Input-channel/row partition | Sum 16×4096 FP32 partial output |
| Residual and normalization | Replicated hidden state | Local after each reduction |
| Vocabulary projection | Vocabulary partition | Combine local greedy maxima; avoid collecting all logits |
| Input embedding | Replicated for simplicity | Small input-token transfer |

This requires **two reductions per layer**, or **64 collectives per decode forward**. The four-way grouping respects the eight KV heads exactly. Token IDs and tiny greedy summaries are included; host computation and fixed collective latency are explicitly separate assumptions.

### 9.2 Theoretical ceiling versus a scheduled estimate

For the streaming policy at mean context 2176, each step requires approximately

\[
D=7{,}504{,}658{,}432+4{,}563{,}402{,}752
=12{,}068{,}061{,}184\ \mathrm{bytes}.
\]

Perfectly distributing that compulsory traffic over four 64-GB/s paths gives

\[
T_{traffic}=D/(4\times64\times10^9)=47.140864\ \mathrm{ms}.
\]

The corresponding optimistic **streaming-traffic ceiling** is **21.21 tokens/s per sequence**, or **339.41 aggregate tokens/s**. This excludes metadata, auxiliary traffic, startup, vector constraints, and all host communication. It is not a demonstrated attainable rate or a universal maximum over compressed/cache-resident algorithms. Persistently retaining some operands across steps would slightly change the traffic lower bound; a larger new cache would change it materially.

Re-evaluating the actual panel schedule at the sharded dimensions gives **55.23 ms**, or **18.11 tokens/s per sequence**, before host-network time. This includes conservative local DRAM/SRAM staging for collectives. The vocabulary dimension is padded to complete 128-column panels. Row-parallel partials are FP32. Staging costs include an extra conservative local copy allowance; a fused row-output staging implementation could avoid part of it.

The distinction matters: 70% of 21.21 is a different target from 70% of 18.11. Neither rate can be guaranteed without a host-link specification.

### 9.3 Host-mediated collective traffic

For an FP32 activation reduction,

\[
X=16\times4096\times4=262{,}144\ \mathrm{bytes}.
\]

Use a concrete host-star algorithm: all four devices send partials to host; host sums them; host returns the sum to all four devices. Each collective transfers `4X+4X=8X` bytes across the aggregate host/device links. Across 64 collectives:

\[
D_{collectives}=64\times8X=134{,}217{,}728\ \mathrm{bytes}=128\ \mathrm{MiB}.
\]

Greedy vocabulary selection contributes 768 bytes per step: four ranks each send 16 `(FP32 score, uint32 token)` pairs, followed by broadcasting 16 uint32 winning IDs to all four ranks. Total modeled host traffic is **134,218,496 bytes per step**. Additional user token-return traffic is negligible at this scale and is not a collective term.

FP16 reductions would approximately halve the main traffic to 64 MiB, but are a different precision choice. Do not use a 64-MiB numerator while claiming FP32 communication.

### 9.4 Bandwidth to reach 70% of the traffic ceiling

The target step time is

\[
T_{target}=T_{traffic}/0.7=67.344091\ \mathrm{ms}.
\]

With `T0=55.230066 ms` for the local four-way schedule, zero fixed host collective latency, and host communication serialized with local work, the available communication time is **12.114025 ms**. Therefore

\[
B_{host,min}=\frac{134{,}218{,}496}{0.067344091-0.055230066}
\approx\boxed{11.08\ \mathrm{GB/s}}.
\]

This is the sum of useful payload bandwidth over **all four device links and both directions**. With balanced symmetric traffic, it corresponds to about **2.77 GB/s per device bidirectionally**, or **1.38 GB/s in each direction per device**, averaged over the communication budget. The host memory subsystem must also perform the reduction reads/writes; that separate bandwidth/CPU capacity is not specified.

Let `lambda` be fixed latency per completed gather/reduce/broadcast collective. A more complete requirement is

\[
B_{host}\geq\frac{D_{host}}{T_{traffic}/0.7-T_0-64\lambda-T_{other\,host}}.
\]

A nonpositive denominator means the target is impossible regardless of bandwidth. With other host time zero, `lambda` must be below roughly **189 microseconds**; useful engineering margins require it to be substantially smaller. Greedy selection has an additional small synchronization not represented by the 64 hidden-state collectives and can be included in `T_other_host`.

Under **perfect overlap**, the much weaker traffic-only necessary condition is approximately **1.99 GB/s**. It is not a sufficient link requirement: the next operation depends on the completed reductions, so the 64 barriers cannot simply disappear. The 11.08-GB/s figure answers a defined serialized schedule, rather than mislabeling a traffic-only average as sufficient.

If “theoretical maximum” instead means the zero-network rate of the modeled schedule, reaching 70% of **18.11 tokens/s** needs approximately **5.67 GB/s** under the same serialized, zero-latency assumptions. Both interpretations are given because the pages do not define that term or specify a host topology.

## 10. Decision record and submission files

| Decision | Basis | Consequence / verification |
|---|---|---|
| Use vector arithmetic as the primary backend | Literal infinite-compute SIMD assumption and A16 operands | Requires explicit local-storage and FP32 support assumptions |
| Retain KV by sequence and KV head | 1 MiB per prefill KV group fits SRAM | Loads K/V once and shares across four query heads |
| Use separate prefill/decode programs | Many prompt queries versus four current queries per KV group | Decode avoids padded array rows and repeated KV reads |
| Use online numerator state | Stable incremental normalization | Numerical comparison against independent full-row attention |
| Charge all visited causal tiles | Masking is not zero-cost arithmetic removal inside a dense tile | Exact tile count 1088 per head |
| Keep full-K dense weight panels | Maximum panel fits with activation/output macrotiles | SRAM capacity assertions for every matrix |
| Count all transformer linears | TTFT concerns the entire model | Dense work dominates prefill despite the attention-focused task |
| Separate ceilings from schedule estimates | Hardware and host contracts are incomplete | Prevents presenting lower bounds as predictions |
| Use tensor parallelism for latency | Shards both repeated weights and KV reads | Two host-mediated reductions per layer |
| Use FP32 partial reductions | More defensible accumulation contract | Explicit 128-MiB collective traffic per forward |

| File | Purpose | How to inspect or run |
|---|---|---|
| `section_2_2_report.md` | Complete answer, assumptions, derivations, design decisions, limitations | Read this report |
| `kernel_pseudocode.txt` | Control scheduler, vector kernels, allocation, dense schedule, optional array mapping | Read as device-independent pseudocode |
| `performance_model.py` | Reproducible cycle/traffic calculations and sensitivity analysis | `python3 performance_model.py --output results.json` |
| `results.json` | Exact numeric output of the model | Compare with the rounded report tables |
| `reference_attention.py` | Independent numerical reference and correctness checks | `python3 reference_attention.py` |
| `README.md` | Run instructions, interpretation, and validation scope | Start here when reproducing |

The reference and model use Python’s standard library only. No device driver, model weights, network access, or external Python packages are needed to reproduce the included calculations and correctness checks. The pseudocode API is abstract because the assignment supplies no ISA or runtime API.

Before claiming deployment readiness, confirm the actual model configuration, 16-bit numerical format, vector register capacity, DMA outstanding-request behavior, host link/latency, and intended scope of the infinite-compute vector assumption. If a systolic-first answer is required, confirm the array arithmetic and accumulation/rounding contract before assigning numerical throughput to it.
