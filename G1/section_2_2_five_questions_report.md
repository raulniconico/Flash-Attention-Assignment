# Section 2.2 — LLM inference design: separate prefill and decode

This revision answers the same five original questions. **Prefill and decode are different kernels with different tensor shapes, SRAM lifetimes, arithmetic reuse, and performance limits.** The revised decode design keeps current-token activations in SRAM and streams only weights and past KV, with explicit writes of newly produced KV. This report supersedes the earlier decode timing and four-device bandwidth estimates. All timings remain conditional analytical results, not accelerator measurements.

## 1. Provide a pseudocode implementation of an optimized flash-attention kernel for this architecture, detailing the SRAM allocation strategy, the tiling strategy and the overall computational dataflow. Keep in mind that this code, once implemented, would be executed on the control CPU.

### Inference phases and the architectural interpretation

Let batch size `B=16`, prompt length `S=2048`, output length `G=256`, hidden size `D=4096`, head dimension `d=128`, query heads `Hq=32`, and KV heads `Hkv=8`. Thus four query heads share each KV head. The named model is taken to have 32 layers and MLP width 14336; these dimensions are explicit model inputs because the attached pages do not enumerate them. They should be checked against the deployed checkpoint. No model weights are needed for this analysis.

| Property | Prefill | One decode forward |
|---|---|---|
| New tokens processed per sequence | All 2048 prompt tokens | One token |
| Rows in a projection/MLP matrix | `B×S=32768` | `B=16` |
| Q per sequence/query head | `2048×128` | `1×128` |
| K/V per sequence/KV head | Produce 2048 entries | Produce and append one entry; read the past cache |
| Attention scores per query head | Causal `2048×2048` logical matrix | `1×C`, where C is the current attended context |
| Query rows sharing one KV head | Four heads, each with many prompt positions | Four current query rows |
| Main reuse opportunity | Reuse weights across many token rows; reuse K/V across query blocks | Reuse each weight across the batch; reuse K/V across the four query heads |
| State in SRAM | Selected head groups and tiled working activations | Current hidden state, residuals, Q/new-KV, MLP work, and staging buffers |
| Metric primarily affected | TTFT | Inter-token latency and interactivity |

**Do not call the batch-16 linears a batch-1 GEMV.** They are small-M GEMMs. Also, the batch dimension cannot be combined arbitrarily inside attention: different sequences have different KV contents. Only the four query heads sharing the same sequence/KV head use one ordinary shared right operand.

Figure 1 supplies the paths: DRAM ↔ DMA ↔ SRAM, and SRAM ↔ each compute unit. This implies explicit staging, on-chip intermediate exchange, and dependency tracking. Its natural compute mapping is array QK/PV and vector masking, softmax and reductions. The accompanying table supplies 64 B/cycle end-to-end DRAM/DMA/SRAM bandwidth and 128 B/cycle SRAM/vector bandwidth. The two 64 B/cycle links are in series, not additive. Separate engines permit a candidate overlapped schedule; asynchronous execution must still be supported. The instructions allow ignoring SRAM bank conflicts and contention.

### Numerical contract and the two implementation paths

The task specifies 16-bit activations but 8-bit array inputs. It does not identify the activation encoding or the array accumulator/rounding semantics. A raw signed-16 output cannot generally contain a full-range INT8 dot product over 128 or 256 terms. Therefore the controller must dispatch a numerically supported operation, rather than silently casting A16 to A8.

- **Architecture-oriented path:** prefill QK/PV and linears use the arrays, with vector softmax/updates. Its runtime depends on an explicitly specified A16-compatible lowering or an authorized approximate quantization contract. An 8-bit approximation is not automatically allowed by the assignment.
- **Precision-preserving numerical reference:** use the memory-only vector CPU for the dot products as well, taking the stated infinite-computation assumption literally. Assume vector support for the 16-bit representation, FP32 running sums, and at least 12 KiB of vector-local working storage. These are explicit assumptions, not conclusions from Figure 1. The numerical schedule estimates in Questions 2–5 use this path. Array throughput bounds are separately labeled and are not mixed into those estimates.

This distinction is essential: a realistic finite-throughput vector core and the stated idealized vector core can have different optimal mappings.

### Prefill design

1. **Perform Q/K/V projections for the prompt.** The large token-row dimension offers substantial weight reuse and enough independent rows to occupy both arrays if the numerical contract permits it. Process the MLP and other projections using SRAM-sized macrotiles; the entire 32768-row hidden state cannot remain in 16 MiB.
2. **Apply RoPE before storing reusable keys.** Store the post-RoPE K values and V values needed by later decoding. Retain the original position association; do not apply RoPE to the cached keys again on every decode step.
3. **Make `(batch item, KV head)` the attention reuse unit.** Its two 16-bit K/V tensors require `2×2048×128×2 = 1 MiB`. Load once and reuse for the four associated query heads.
4. **Tile the query and key dimensions.** Use `Br=32` on the 32×16 array and `Br=16` on the 16×16 array, with `Bc=128`. Each QK block needs eight 16-key output-column tiles; each PV block needs eight 16-feature output-column tiles. Reduction lengths are 128, within the specified maximum of 256. Assign independent query blocks dynamically; do not permanently dedicate one array to QK and the other to PV.
5. **Use causal online softmax.** Skip key blocks wholly above the causal diagonal; mask future elements in partially valid blocks. Keep FP32 running maxima, denominators and numerators. Never materialize the full attention matrix in DRAM.
6. **Retain final prompt K/V in DRAM.** The next phase requires the cache from every layer, even though only the last prompt position needs the vocabulary projection for the first generated token.

The numerical vector reference uses `Br=16, Bc=128` throughout. It retains an 8 KiB FP32 output accumulator during each matrix microkernel. Its reported runtime is for this concrete schedule, not the array schedule above.

| Prefill SRAM allocation | Size |
|---|---:|
| Bundle 0: one KV group | 1 MiB |
| Bundle 0: four query heads | 2 MiB |
| Bundle 0: four output heads, stored at 16 bits | 2 MiB |
| Bundle 1: same buffers, for overlap | 5 MiB |
| FP32 tile work, descriptors, alignment | 1 MiB |
| Spare | 5 MiB |
| Total | 16 MiB |

The dense and attention phases reuse the same physical SRAM rather than allocating their separate ledgers simultaneously. A dense down-projection macrotile with 256 token rows needs 7 MiB input, 3.5 MiB double-buffered weight panels, 2 MiB output, and 1 MiB reserve: 13.5 MiB. The calculator asserts the relevant dense capacity bound.

### Decode design

**Decode does not recompute the prompt’s Q/K/V, outputs, or MLPs.** For each layer, it processes only the current token of each sequence and consults that layer’s stored KV history.

1. Keep the current `16×4096` hidden tensor in SRAM: only **128 KiB** at 16 bits. Keep a separate residual buffer. Stream the required weight panels and produce the current Q/K/V directly into SRAM.
2. Apply RoPE using the current absolute position. The newly generated K/V together occupy **64 KiB per layer across batch 16 and eight KV heads**.
3. Process attention by `(batch item, KV head)`. Load the **past** K/V once, share it across its four current query heads, include the new K/V already in SRAM, and update a four-row online-softmax state.
4. Use a decode-specific vector program: four Q rows, four FP32 output numerators, maxima/sums, and a 4×128 temporary. A 12 KiB local working budget is sufficient for the stated design. No prompt-sized score tile or query buffer is needed.
5. Write the new KV entry once to persistent DRAM cache. Do not write back the entire old cache, and do not reload the newly produced entry merely because it has been appended to DRAM.
6. Keep attention output in SRAM for the output projection and residual. Keep the small MLP activations in SRAM as well. A `16×14336` gate or up tensor is **448 KiB**; its buffer can be reused for the gated result. Only weights, past KV, and persistent KV appends need the large external-memory path in the steady-state design.

For contiguous KV transfers, use a head-group layout `[batch, kv_head, token, K_or_V, channel]`. Each token’s K/V pair is 512 contiguous bytes. One append descriptor per group is charged; a single global contiguous append for all head groups would be an incorrect assumption for this layout. A layout conversion from projection outputs must be emitted by the epilogue or explicitly accounted for. The numerical example assumes prepacked weight/cache layouts and charges the current-KV SRAM packing in decode.

A conservative decode SRAM ledger fits below 16 MiB: up to 3.5 MiB for two full-K weight panels, about 2.25 MiB for two end-of-generation KV staging groups, less than 2 MiB for hidden/residual/Q/MLP and output work, plus 1 MiB reserve. The vocabulary head runs after those attention/MLP buffers can be released; even a full batch of 16-bit logits is only about 3.91 MiB. Greedy selection can instead be fused with vocabulary panels.

Why prefer this decode attention kernel? Four useful rows occupy only **25%** of a 16-row array or **12.5%** of a 32-row array if padding is required. It is not valid to fill the spare rows with unrelated sequences while pretending they share the same K/V matrix. For linears, batch 16 fills the 16×16 array but only half the 32×16 array unless it supports independent partitions or another valid packing scheme.

### Controller pseudocode: distinct entry points and KV lifecycle

The control CPU executes the orchestration below. `VECTOR_*` and `MATRIX_*` launch work on compute units; arithmetic inside those programs does not execute on the control CPU. Array use is guarded by the numerical contract. All data read by compute units are in SRAM.

```text
INFERENCE(prompt[16,2048], output_tokens=256):
    kv_length = 2048
    hidden = PREFILL_EMBED(prompt)
    for layer in 0..31:
        hidden = PREFILL_LAYER(layer, hidden, prompt_positions=0..2047)
    y1 = VOCAB_AND_SAMPLE(hidden[:,2047,:])
    emit y1

    current = y1
    for output_index in 2..256:
        position = kv_length
        hidden_sram = EMBED_CURRENT_TOKEN(current)
        for layer in 0..31:
            hidden_sram = DECODE_LAYER(layer, hidden_sram, position)
        current = VOCAB_AND_SAMPLE(hidden_sram)
        emit current
        kv_length += 1

PREFILL_LAYER(layer, hidden_dram, prompt_positions):
    # Dense macrotiles: M=16*2048, with explicit SRAM staging.
    q, k, v = PROJECT_PROMPT_AND_APPLY_ROPE(layer, hidden_dram)
    STORE_PROMPT_KV(layer, k, v)  # all prompt positions, every layer
    for each (batch_item, kv_head), double-buffered:
        LOAD_GROUP_Q_AND_KV_INTO_FREE_SRAM_BUNDLE()
        WAIT_FOR_INPUT_DMA()
        for each of the 4 associated query heads:
            for query block i:
                initialize m=-inf, ell=0, U=0 in SRAM
                for key block j with any key_position <= query_position:
                    S = MATRIX_QK_SUPPORTED_BACKEND(Q_i, K_j) / sqrt(128)
                    VECTOR_MASK_AND_ONLINE_UPDATE(S, m, ell, U)
                    # Online update returns unnormalized P and rescales U.
                    U += MATRIX_PV_SUPPORTED_BACKEND(P, V_j)
                VECTOR_NORMALIZE_AND_CAST_OUTPUT(U/ell)
        STORE_COMPLETED_OUTPUT_BUNDLE()
    return OUTPUT_PROJECTION_RESIDUAL_MLP_TILED(layer)

DECODE_LAYER(layer, hidden_sram, position):
    residual_sram = hidden_sram
    q_sram, k_new, v_new = PROJECT_CURRENT_AND_ROPE(hidden_sram, position)
    for each (batch_item, kv_head), double-buffered:
        LOAD_PAST_KV_ONLY(layer, positions=0..position-1)
        WAIT_FOR_INPUT_DMA()
        PLACE_NEW_KV_FROM_SRAM_AFTER_PAST_KV(k_new, v_new)
        VECTOR_LAUNCH_DECODE_GROUP(
            queries=4 associated current query heads,
            keys_values=positions 0..position,
            block_keys=128,
            state=(m[4], ell[4], U[4,128]))
        STORE_NEW_KV_PAIR_TO_DRAM(position)  # exactly 512 bytes per group
        KEEP_GROUP_ATTENTION_OUTPUT_IN_SRAM()
    WAIT_FOR_OUTPUT_AND_CACHE_APPEND_COMPLETIONS()
    hidden_sram = OUTPUT_PROJECTION_AND_RESIDUAL_IN_SRAM(residual_sram)
    return NORM_GATED_MLP_AND_RESIDUAL_IN_SRAM(hidden_sram)

ONLINE_UPDATE(scores, values, m, ell, U):
    if a row has no valid keys in this block: leave its state unchanged
    m_new = max(m, rowmax(scores))
    alpha = 0 if ell == 0 else exp(m - m_new)
    P = exp(scores - m_new)
    ell = alpha*ell + rowsum(P)
    U = alpha*U + P*values
    m = m_new
    # After all key blocks: output = U / ell
```

The last procedure expresses the mathematics; in the prefill schedule its softmax/rescaling and PV portions are dispatched separately. A bundle progresses through `FREE → LOADING → READY → COMPUTING → STORING → FREE`. Never overwrite a buffer until all consumers and its outstanding writes have completed. Only independent jobs overlap; the score → softmax → PV dependency cannot be overlapped with itself.

The online-softmax recurrence is the standard tiled attention technique described in [FlashAttention](https://arxiv.org/abs/2205.14135) and [FlashAttention-2](https://arxiv.org/abs/2307.08691). The supplied numerical reference previously passed 19 comparisons against full-row attention, including causal/noncausal, GQA, decode, tails and large logits. That validates the recurrence/indexing in Python floating point, not hardware execution or the accuracy of an unspecified A8 conversion.

## 2. Identify and explain the bottleneck of this accelerator architecture for the flash-attention kernel, i.e. the parameter with the largest elasticity with respect to the required time for this kernel to complete.

### Prefill bottleneck

For an array implementation, prefill exposes many token rows and substantial reuse. Candidate limits are array execution, the 8 B/cycle array output ports, vector processing of intermediate scores, and launch overhead. The ports imply a minimum output-drain time of **64 cycles** for 16×16 and **128 cycles** for 32×16 at 16-bit output. Whether that is the actual bottleneck depends on compute initiation and overlap. Figure 1 alone does not establish the largest elasticity.

In the specified **memory-only vector reference**, prefill attention is instead dominated by SRAM/vector traffic: each visited 16×128 query/key tile transfers 135,424 bytes across that interface. A resident head group needs about 4.687 million vector cycles against only 83,070 DMA cycles. Thus faster DRAM has little influence on this particular prefill kernel; faster SRAM/vector access has nearly proportional benefit.

### Decode bottleneck

With the redesigned vector kernel, every past K/V element is read once per sequence/KV group and reused across four query heads. Across all layers at mean context 2176, past-KV reads plus new-KV writes total **4.25 GiB per forward**. The dense stages stream about **7.505 GB of weights**. These two contributions must be counted separately: attention is not the whole inference step, and the weights are actually the larger external-memory contribution here.

Attention arithmetic intensity is approximately **2 MAC per external-memory byte**, or 4 FLOPs/byte, after GQA reuse. Decode linears have approximately **16 MAC per weight byte**, before other traffic. Prefill has much greater reuse across token rows, although limited SRAM and tiling cause some weight rereads.

**Do not assume all decode is memory-bound on every implementation of this machine.** If each array PE performs one MAC/cycle and ordinary decode linears use only 16 useful rows on either array, their combined useful dense capacity is at most 512 GMAC/s. The required 120.075 billion dense MACs then take at least **234.52 ms**, already greater than the roughly **117.26 ms** raw weight-read time. On that conditional array mapping, dense decode can be compute-bound. The memory-bound conclusion below is specific to the idealized vector implementation, where finite arithmetic throughput is deliberately absent by specification.

### Elasticity, evaluated separately

Use `E_x = -d ln(T)/d ln(x)` for a capacity parameter. Positive values indicate runtime reduction when capacity grows. The updated calculator uses a 1% finite increase.

| Capacity increased | Prefill attention | Decode attention | Complete decode forward |
|---|---:|---:|---:|
| SRAM/vector bandwidth | 0.9998 | 0.0037 | 0.0131 |
| Both DRAM/DMA and DMA/SRAM bandwidths together | 0.00013 | 0.9528 | 0.9446 |
| DRAM/DMA alone, with the other link fixed at 64 B/cycle | 0 | 0 | 0 |
| DMA/SRAM alone, with the other link fixed at 64 B/cycle | 0 | 0 | 0 |

For this reference: **prefill attention is SRAM/vector-bandwidth limited; decode attention and the complete decode forward are primarily external-memory-path limited.** The two equal 64 B/cycle series links are a joint bottleneck. Increasing one alone leaves the other limiting; the `min(B1,B2)` function has a kink at equality. Therefore there is no unique smooth “largest scalar elasticity” for either individual link at this point.

These are kernel/runtime conclusions, not claims that prefill and decode share one universal bottleneck. TTFT also includes all transformer linears and non-attention operations.

## 3. Provide estimations of the TTFT (Time to first token) and the interactivity (token generation rate) for this workload and architecture. These numbers must be justified by a performance model which takes into account the architectural constants detailed above.

### Token accounting and metric definitions

Prefill processes the 2048 prompt tokens and supplies logits for **generated token 1**. The remaining **255** forwards consume generated tokens one at a time and produce tokens 2 through 256.

| Event | Query position, zero-based | Attended context length |
|---|---:|---:|
| Last prefill query | 2047 | 2048 |
| First decode forward, consuming generated token 1 | 2048 | 2049 |
| Last decode forward, consuming generated token 255 | 2302 | 2303 |

The arithmetic mean decode context is 2176. Reserve cache capacity through 2304 if convenient, but do not charge 256 decode forwards after already generating the first token during prefill. Standard TTFT is prompt processing plus first-token selection/return, not the time to generate all 256 tokens.

For a synchronized batch of 16, interactivity per sequence is `1/T_step`; aggregate rate is `16/T_step`. Increasing aggregate throughput is not equivalent to increasing per-user interactivity.

### Model dimensions, memory and work

The seven transformer weight matrices per layer have input/output dimensions `(4096,4096)`, `(4096,1024)`, `(4096,1024)`, `(4096,4096)`, `(4096,14336)`, `(4096,14336)` and `(14336,4096)`. They contain 218,103,808 weights per layer. Across 32 layers plus the `4096×128256` vocabulary head, each decode step uses **7,504,658,432 W8 weight bytes**, excluding the explicitly charged per-output-channel scale metadata. Input embeddings bring the large weight tensors to about 8.030 GB total.

| Quantity | Prefill | Mean decode forward |
|---|---:|---:|
| Transformer dense MACs | 228.698 trillion | Included in 120.075 billion below |
| Dense MACs including vocabulary head | Add only 16 last-position vocabulary projections | 120.075 billion |
| Attention MACs | 8.800 trillion, ideal causal count | 9.127 billion |
| KV storage | 4 GiB after prompt | About 4.25 GiB at mean context |

With one MAC defined as a multiply plus accumulate:

`MAC_prefill_attention = L × B × Hq × d × S × (S+1)`.

`MAC_decode_attention(C) = 2 × L × B × Hq × d × C`.

`KV_bytes(C) = 2 × L × B × Hkv × d × C × 2`.

Use KV heads, not query heads, in the cache formula. The full cache and weights fit 64 GiB DRAM, but neither fits 16 MiB SRAM. Keeping current activations in SRAM during decode does not mean keeping the full cache there.

### Architectural costs and overlap

At 1 GHz, one cycle is 1 ns. For the conservative aggregate-bidirectional interpretation:

- `b = min(DRAM_DMA, DMA_SRAM) = 64 bytes/cycle`.
- `read(n) = 200 configuration + 200 first-byte + ceil(n/b)` cycles.
- `write(n) = 200 configuration + 150 first-byte + ceil(n/b)` cycles.
- `vector(n) = 300 launch + ceil(n/128)` cycles for the stated memory-only core.

A double-buffered stream uses a maximum for overlapping independent stages, plus startup/drain. Dependent layers and suboperations are summed. Do not add the two serial-link bandwidths, and do not hide all end-to-end work inside one global maximum.

Prefill dense work uses 256-row macrotiles and 16×128 vector output microtiles. A full-K weight panel and its scales are loaded while the preceding panel executes. A macro with `r` token rows, reduction K, and N output columns uses:

`W_panel_bytes = 128K + 512`.

`vector_panel_bytes = (r/16)128K + 2rK + 2r128 + 512`.

`T_macro = read(2rK) + R + V + (N/128 - 1)max(R,V) + write(2rN)`.

Here `R=read(W_panel_bytes)` and `V=vector(vector_panel_bytes)`. FP32 tensor-parallel partials replace the final 2-byte output width with 4 bytes. This charges weight rereads induced by the tiling; it does not assume the full model is read once for prefill.

For **decode**, r is 16 and the current activation/output tensors stay resident: the first input-read and final output-write terms are removed. Weight-panel transfers and all modeled vector accesses remain. Norm/residual/MLP activation passes are charged SRAM/vector traffic, not unnecessary DRAM round-trips.

For each decode KV group, with C attended keys:

`R = read((C-1) × 512)` — past cache only.

`W = write(512)` — exactly one new contiguous K/V pair.

`V = vector(C×512 + query/output bytes + current-KV packing bytes)`.

Double buffering overlaps `R+W` with independent vector jobs. The model uses a conservative per-layer boundary allowance. This representation explicitly charges small append descriptors and avoids the earlier assumption that all head-group appends are one contiguous DMA.

### Revised numerical estimates

**These are the precision-preserving memory-only-vector scenario, with prepacked layouts, W8 per-channel scales, FP32 state, at least 12 KiB vector-local storage, preloaded weights and greedy sampling.** They are not the measured performance of an unspecified array lowering.

| Component | Warm prefill / first token | Mean decode forward |
|---|---:|---:|
| Transformer linears | 143.7458 s | Included in dense total |
| Vocabulary head / selection | 0.0088 s | Included in dense total |
| Dense total during decode | — | 123.7255 ms |
| Attention, including decode cache appends | 19.3512 s | 75.2451 ms |
| Auxiliary operations/transfers | 4.6349 s | 0.8041 ms |
| **Total** | **167.7407 s** | **199.7747 ms** |

| Decode context | Latency | Per-sequence interactivity | Aggregate throughput |
|---|---:|---:|---:|
| 2049 | 195.56 ms | 5.11 tokens/s | 81.81 tokens/s |
| 2176 | 199.77 ms | 5.01 tokens/s | 80.09 tokens/s |
| 2303 | 203.99 ms | 4.90 tokens/s | 78.44 tokens/s |

The 255 decode forwards take **50.94 s**; the complete 256-token response takes approximately **218.68 s including warm prefill**. This includes neither unknown host preprocessing/queueing nor cold model loading.

The revised decode schedule reduces the earlier 204.84 ms mean to **199.77 ms**. The change is modest because eliminating activation spills does not eliminate the much larger weight and KV streams. Prefill remains **167.74 s** in this numerical reference: its schedule has not been changed merely to force a different answer.

For an explicitly different, conventional **one-MAC/PE/cycle array scenario**, the 768-PE aggregate gives an optimistic prefill arithmetic lower bound of about **297.78 s for transformer linears plus 11.46 s for attention**, before launch, port, precision-lowering and other costs. This is not a second estimate for the same vector program. The large difference explains why a hardware-accurate answer requires clarifying the vector arithmetic contract and the array A16 lowering. Likewise, the 234.52-ms decode dense bound in Question 2 cannot be combined with an assumed memory-only dense time.

Observed TTFT must include unknown host terms:

`TTFT = preprocessing + input transfer + queueing + device prefill/head + token return`.

Cold start additionally requires transferring the weights over an unspecified host link. Do not invent that bandwidth or call the warm result a cold-start estimate.

### Reproducible calculator

The complete revised standard-library calculator is embedded below. Save this block as `model.py` and run `python3 model.py --output results.json`. It contains the separate prefill/decode schedules, SRAM checks, context sweep, four-device communication calculations and sensitivity analysis. It is an analytical calculator, not a cycle-accurate simulator.

```python
#!/usr/bin/env python3
"""Conditional analytical model; not a cycle-accurate simulator.
Run: python3 performance_model.py --output results.json
All bandwidths are aggregate bytes/cycle. A MAC is one multiply-add.
Inference phase-aware revision: SRAM-resident decode activations and per-group KV appends.
Primary numerical backend: literal memory-only vector CPU, 16-bit stored activations,
8-bit weights, FP32 running state, >=12 KiB vector-local working storage.
"""
from dataclasses import dataclass, replace
import argparse, json, math

@dataclass(frozen=True)
class Hardware:
    hz: float = 1e9
    dram_bw: float = 64.0
    dma_sram_bw: float = 64.0
    vector_bw: float = 128.0
    dma_setup: float = 200.0
    read_latency: float = 200.0
    write_latency: float = 150.0
    vector_launch: float = 300.0
    @property
    def path_bw(self):
        return min(self.dram_bw, self.dma_sram_bw)
    def read(self, n):
        return self.dma_setup + self.read_latency + math.ceil(n/self.path_bw)
    def write(self, n):
        return self.dma_setup + self.write_latency + math.ceil(n/self.path_bw)
    def vector(self, n):
        return self.vector_launch + math.ceil(n/self.vector_bw)

L, D, F, HQ, HKV, HD, VOCAB = 32, 4096, 14336, 32, 8, 128, 128256
BATCH, PROMPT, OUTPUT = 16, 2048, 256

def matrices(tp=1):
    assert HQ % tp == HKV % tp == F % tp == 0
    return [('q', D, D//tp), ('k', D, HKV*HD//tp),
            ('v', D, HKV*HD//tp), ('o', D//tp, D),
            ('gate', D, F//tp), ('up', D, F//tp),
            ('down', F//tp, D)]

def dense_cycles(m, k, n, hw, out_bytes=2, resident=False):
    """256-row DRAM macrotiles, 16x128 register microtiles.
    Keep full A macrotile in SRAM; double-buffer full-K weight panels.
    For each macro, serialize input/output DMA with its weight pipeline.
    Quantization scale: one FP32 number/output column; read once per panel.
    """
    assert n % 128 == 0 and m % 16 == 0
    total = 0.0
    panels = n//128
    for i in range(0, m, 256):
        rows = min(256, m-i)
        # Worst SRAM: A + two weight panels + full output + workspace.
        allocated = rows*k*2 + 2*k*128 + rows*n*out_bytes + 1024*1024
        assert allocated <= 16*1024*1024, (m,k,n,allocated)
        r = hw.read(k*128 + 128*4)
        # W reread per 16 rows; A read per output panel; FP16 output write.
        io = (rows//16)*k*128 + rows*k*2 + rows*128*out_bytes + 128*4
        v = hw.vector(io)
        total += 0 if resident else hw.read(rows*k*2)
        total += r + v + (panels-1)*max(r,v)
        total += 0 if resident else hw.write(rows*n*out_bytes)
    return total

def elementwise_cycles(m, tp, hw):
    # Conservative unfused traffic budget for 2 RMSNorm, 2 residual adds,
    # RoPE on local Q/K, and local SiLU*gate. 5 launched macro programs.
    traffic = m*(20*D + 4*(D+HKV*HD)/tp + 6*F/tp)
    # Budget an entire read/write DMA pass and vector pass serially.
    # 5 read/write pairs cover the aggregate bytes; no compute-time term.
    dma = traffic/hw.path_bw + 5*(2*hw.dma_setup+hw.read_latency+hw.write_latency)
    vec = traffic/hw.vector_bw + 5*hw.vector_launch
    return dma+vec

def prefill_attention_cycles(hw, tp=1):
    groups = BATCH*HKV//tp
    query_blocks = PROMPT//16
    visited = sum(math.ceil((i+16)/128) for i in range(0,PROMPT,16))
    # One QK tile, online update, then PV tile. FP32 S/P/O, FP16 Q/K/V.
    qk = 16*128*2 + 128*128*2 + 16*128*4
    online = 2*16*128*4 + 2*16*128*4 + 2*16*2*4
    pv = 16*128*4 + 128*128*2 + 2*16*128*4
    tile_bytes = qk+online+pv
    # Initialize O,m,l; final O read + FP16 write for every query block.
    boundary_bytes = 16*128*4 + 16*2*4 + 16*128*6
    vec = hw.vector(4*(visited*tile_bytes+query_blocks*boundary_bytes))
    r = hw.read(2*PROMPT*HD*2) + hw.read(4*PROMPT*HD*2)
    w = hw.write(4*PROMPT*HD*2)
    # Double-buffer full head-group bundles; conservative boundary allowance.
    layer = groups*max(r+w,vec)+r+vec+w
    return L*layer, {'visited_tiles_per_head':visited,'tile_vector_bytes':tile_bytes,
                     'vector_cycles_per_group':vec,'dma_cycles_per_group':r+w}

def decode_attention_cycles(context, tp, hw):
    groups = BATCH*HKV//tp
    kv = 2*context*HD*2
    qout = 4*HD*2
    # Four Q heads share K/V. Q,m,l,O stay local during the full KV sweep.
    # Cache layout is [batch, kv_head, token, K_or_V, channel].
    # Past KV is read once; the current token is already in SRAM.
    r = hw.read(kv-2*HD*2)
    w = hw.write(2*HD*2)  # one contiguous 512-byte append per group
    v = hw.vector(kv+2*qout+2*(2*HD*2))  # includes current-KV SRAM packing
    return L*(groups*max(r+w,v)+r+v+w)

def reduction_staging_cycles(hw, payload_bytes=4):
    x = BATCH*D*payload_bytes
    # Per rank, two collectives/layer. Outgoing SRAM->DRAM and reverse.
    return 2*L*(hw.write(x)+hw.read(x))

def head_and_sampling_cycles(hw, tp=1, resident=False):
    # 128256/4 is not divisible by 128: pad last vocabulary panel.
    n = math.ceil(VOCAB/tp/128)*128
    c = dense_cycles(BATCH,D,n,hw,resident=resident)
    # Read local FP16 logits once for exact local greedy argmax.
    c += (0 if resident else hw.read(BATCH*n*2))+hw.vector(BATCH*n*2)
    return c

def decode(context, hw=Hardware(), tp=1, batch=BATCH):
    assert batch == BATCH, 'This TP model holds global batch at 16.'
    dense = L*sum(dense_cycles(BATCH,k,n,hw,4 if tp>1 and name in ('o','down') else 2,resident=True) for name,k,n in matrices(tp))
    dense += head_and_sampling_cycles(hw,tp,resident=True)
    attn = decode_attention_cycles(context,tp,hw)
    aux_bytes = BATCH*(20*D + 4*(D+HKV*HD)/tp + 6*F/tp)
    aux = L*(aux_bytes/hw.vector_bw + 5*hw.vector_launch)
    # KV append is charged per group in attention; embedding is loaded once.
    cache_write = 0  # per-group appends already charged in attention
    embedding = hw.read(BATCH*D*2)
    staging = reduction_staging_cycles(hw) if tp>1 else 0
    cycles = dense+attn+aux+cache_write+embedding+staging
    return {'dense_s':dense/hw.hz,'attention_s':attn/hw.hz,'auxiliary_s':(aux+cache_write+embedding)/hw.hz,
            'collective_staging_s':staging/hw.hz,'step_s':cycles/hw.hz,
            'per_sequence_tokens_s':hw.hz/cycles,'aggregate_tokens_s':BATCH*hw.hz/cycles}

def prefill(hw=Hardware(), tp=1):
    m = BATCH*PROMPT
    dense = L*sum(dense_cycles(m,k,n,hw,4 if tp>1 and name in ('o','down') else 2) for name,k,n in matrices(tp))
    head = head_and_sampling_cycles(hw,tp)
    attn, details = prefill_attention_cycles(hw,tp)
    aux = L*elementwise_cycles(m,tp,hw)
    # QKV projections already write K/V to DRAM; no second cache write here.
    embed = hw.read(m*D*2)
    staging = 2*L*(hw.write(m*D*4)+hw.read(m*D*4)) if tp>1 else 0
    total = dense+head+attn+aux+embed+staging
    return {'dense_s':dense/hw.hz,'head_and_sampling_s':head/hw.hz,
            'attention_s':attn/hw.hz,'auxiliary_s':(aux+embed)/hw.hz,
            'collective_staging_s':staging/hw.hz,
            'warm_ttft_s':total/hw.hz,'attention_details':details}

def elasticity(fn, hw, field, delta=0.01):
    base = fn(hw)
    modified = fn(replace(hw,**{field:getattr(hw,field)*(1+delta)}))
    return -math.log(modified/base)/math.log(1+delta)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--output'); a=p.parse_args()
    hw=Hardware()
    weight_layer=sum(k*n for _,k,n in matrices())
    weights=L*weight_layer+D*VOCAB
    contexts=list(range(PROMPT+1,PROMPT+OUTPUT)) # 255 decode forwards
    dec=[decode(c,hw) for c in contexts]
    tp=[decode(c,hw,4) for c in contexts]
    avg=sum(x['step_s'] for x in dec)/len(dec)
    avg4=sum(x['step_s'] for x in tp)/len(tp)
    # Host reduction: 4 gather + 4 scatter transfers per collective.
    comm=2*L*8*BATCH*D*4
    # Greedy vocab reduction: FP32 score + uint32 token ID per rank,
    # then broadcast uint32 winners to all ranks: 768 bytes/step.
    comm+=4*BATCH*8+4*BATCH*4
    raw_kv=2*L*BATCH*HKV*HD*2176*2
    floor=(weights+raw_kv)/(4*hw.path_bw*hw.hz)
    report={
      'assumptions':{'backend':'memory-only vector CPU; >=12 KiB local working storage',
                     'sampling':'greedy','resident_model':True,'reduction_payload_bytes':4,
                     'host_collective_latency_s':0,'quant_scale':'FP32 per output column',
                     'array_compute_timing':'not needed by primary backend'},
      'counts':{'layer_linear_weights':weight_layer,'streamed_weight_bytes_per_step':weights,
                'total_weight_bytes_including_input_embedding':weights+D*VOCAB,
                'kv_bytes_at_2048':2*L*BATCH*HKV*HD*2048*2,
                'kv_bytes_at_2304':2*L*BATCH*HKV*HD*2304*2,
                'prefill_dense_macs':BATCH*PROMPT*L*weight_layer,
                'prefill_causal_attention_macs':L*BATCH*HQ*HD*PROMPT*(PROMPT+1),
                'decode_dense_macs':BATCH*weights,'decode_attention_macs_at_2176':2*L*BATCH*HQ*HD*2176},
      'prefill':prefill(hw),
      'prefill_four_way_local':prefill(hw,4),
      'decode_first':dec[0],'decode_average_context':decode(2176,hw),'decode_last':dec[-1],
      'generation':{'decode_forwards':len(contexts),'decode_total_s':sum(x['step_s'] for x in dec),
                    'average_step_s':avg,'per_sequence_tokens_s':1/avg,'aggregate_tokens_s':BATCH/avg},
      'four_way':{'average_step_excluding_host_network_s':avg4,'conditional_per_sequence_tokens_s':1/avg4,
                  'conditional_aggregate_tokens_s':BATCH/avg4,
                  'traffic_floor_s':floor,'traffic_ceiling_per_sequence_tokens_s':1/floor,
                  'traffic_ceiling_aggregate_tokens_s':BATCH/floor,
                  'host_aggregate_bytes_per_step':comm,
                  'prefill_host_bytes':2*L*8*BATCH*PROMPT*D*4+768,
                  'host_GB_s_70pct_schedule_serial':comm/((1/0.7-1)*avg4)/1e9,
                  'host_GB_s_70pct_traffic_ceiling_serial':comm/(floor/0.7-avg4)/1e9 if floor/0.7>avg4 else None,
                  'host_GB_s_70pct_traffic_ceiling_perfect_overlap':comm/(floor/0.7)/1e9},
      'sensitivity':{},
      'improvements':{'double_memory_path':decode(2176,replace(hw,dram_bw=128,dma_sram_bw=128)),
                      'double_vector_bandwidth':decode(2176,replace(hw,vector_bw=256)),
                      'ideal_dma_descriptor_and_latency_hiding':decode(2176,replace(hw,dma_setup=0,read_latency=0,write_latency=0))}}
    for name,fn in [('prefill_attention',lambda h:prefill_attention_cycles(h)[0]/h.hz),
                    ('decode_attention',lambda h:decode_attention_cycles(2176,1,h)/h.hz),
                    ('decode_total',lambda h:decode(2176,h)['step_s'])]:
        report['sensitivity'][name]={field:elasticity(fn,hw,field) for field in
              ['dram_bw','dma_sram_bw','vector_bw','dma_setup','vector_launch']}
        both=replace(hw,dram_bw=64*1.01,dma_sram_bw=64*1.01)
        report['sensitivity'][name]['both_memory_links']= -math.log(fn(both)/fn(hw))/math.log(1.01)
    out=json.dumps(report,indent=2)
    if a.output:
        with open(a.output,'w') as f:f.write(out+'\n')
    print(out)

if __name__=='__main__':main()
```

## 4. Suggest 2 architectural improvements to the accelerator that would improve the interactivity.

### Improvement 1: widen the complete memory path for decode

Increase both DRAM–DMA and DMA–SRAM from 512 to 1024 bits/cycle. One widened link alone cannot improve a path still limited by the other 512-bit link. This directly accelerates repeated weight reads and past-KV scans.

In the revised vector-reference model, mean decode latency becomes **118.73 ms**, giving **8.42 tokens/s per sequence**, approximately **1.68×** the baseline. The benefit is below 2× because other work and SRAM/vector limits remain. Prefill attention benefits little in this reference because its on-chip traffic dominates.

### Improvement 2: autonomous DMA descriptor execution with latency hiding

Provide a hardware descriptor queue and multiple outstanding transfers so the control CPU can enqueue weight-panel reads, past-KV reads and small persistent-cache appends without paying an exposed configuration/first-byte penalty for every operation. This is especially relevant to decode’s repeated group-level operations; it does not require reducing activation precision.

An optimistic calculation fully amortizing recurring DMA setup/first-byte costs gives **191.98 ms** and **5.21 tokens/s per sequence**, approximately **4.1%** better than baseline. Actual gains will be smaller if first/last transfers or limited outstanding capacity expose latency. This is a modest improvement, correctly reflecting that byte throughput is the larger limit.

Both recommendations target **interactivity during decoding**. Larger prefill tiles or higher prefill matrix throughput primarily improve TTFT. If the actual implementation uses finite-throughput vector arithmetic or array-based A16 emulation, recompute the ranking: the dense compute limit identified in Question 2 may then make native A16 support and flexible array partitioning more valuable than the second improvement. That is a change of numerical/hardware model, not a result established by the current timings.

## 5. Suppose that 4 such accelerators can be connected to a single host CPU.

### How would you parallelize the computation of this workload across the 4 accelerators?

Use four-way tensor parallelism when the objective is the latency of each of the 16 sequences. Keep the global batch on all four ranks, shard Q into eight query heads and K/V into two KV heads per rank, and shard the MLP intermediate dimension into 3584 channels per rank. Preserve the four-to-one Q/KV grouping locally.

Shard Q/K/V and gate/up by output channels; shard attention-output and MLP-down by input channels and reduce their partial outputs. This requires two hidden-state reductions per layer. Shard the vocabulary projection and combine local greedy maxima instead of moving the complete logits. Residual/norm state remains replicated after reduction.

| Four-device consideration | Prefill | Decode |
|---|---|---|
| Local attention | Many query positions, sharded heads | Four current Q heads per local KV group |
| Local KV cache | Build the prompt shard | Read/update only the local shard |
| Hidden-state reduction shape | `32768×4096` | `16×4096` |
| FP32 reduction payload per rank | 512 MiB | 256 KiB |
| Scheduling focus | Large local work; chunk communication to limit buffers | Low-latency barriers on every generated token |

Prefill reductions cannot require their full payload to fit in 16 MiB: stream them in SRAM-sized chunks through device DRAM/host. Chunking reduces peak buffer demand, not the total payload for this particular collective algorithm. The numerical model optimistically treats the host/network transfer as a bulk term; fine-grained chunk overhead would need an additional host runtime specification.

Data parallelism is an alternative for throughput: four replicas can process four independent batches of four. It still streams a full model per device and does not imply a fourfold per-sequence interactivity gain. Pipeline parallelism can improve utilization across microbatches, but each autoregressive sequence must traverse the layer stages in order before its next token is available. Neither should be substituted for tensor parallelism while claiming its latency scaling.

### What is the theoretical maximum interactivity that can be achieved by this system?

Define a conditional streaming-traffic ceiling for **decode**, not for prefill. At mean context C=2176:

- Streamed weights: 7,504,658,432 bytes.
- Past-KV reads plus new-KV writes: 4,563,402,752 bytes.
- Total ideal bulk traffic: 12,068,061,184 bytes.

Past-cache reads cover C−1 entries and the append covers one entry, so their sum equals the C-entry cache size; the current entry is not redundantly read from DRAM.

Perfectly dividing this traffic across four 64-GB/s paths gives `T_floor=47.140864 ms`, hence **21.21 tokens/s per sequence**, or **339.41 aggregate tokens/s**. This is an optimistic ceiling for exhaustive, uncompressed streaming with no persistent cross-step operand cache, ignoring metadata, startup, network and remaining compute limits. The specification is insufficient for a universal achievable maximum independent of implementation.

The revised schedule, evaluated at sharded dimensions with conservative local reduction staging, gives **52.760244 ms**, or **18.95 tokens/s per sequence**, before host-network time. This is the practical analytical scenario; it is distinct from the 21.21 traffic ceiling.

For completeness, the four-device **prefill** local estimate is about **46.06 s before host-network time**. It is not an interactivity figure and must not be converted into a decode rate. The large prefill collective traffic described next can materially increase TTFT.

### What is the minimum required data bandwidth between the host CPU and the 4 accelerators to achieve 70% of the theoretical maximum interactivity?

Specify a host-star gather/reduce/broadcast algorithm and FP32 partial sums. A decode hidden tensor has `X=16×4096×4=262144 bytes`. Four outgoing partials and four returned sums transfer `8X` bytes per collective. With 64 collectives, that is **128 MiB per decode forward**. Add 768 bytes for greedy vocabulary selection and winning-token broadcast, giving **134,218,496 bytes**.

For 70% of the **21.21-token/s streaming ceiling**, the target period is `47.140864/0.7=67.344091 ms`. The local schedule uses 52.760244 ms, leaving 14.583847 ms for serialized host communication. At zero fixed collective latency:

`B_host >= 134218496 / (0.067344091 - 0.052760244)`

`B_host >= 9.203 GB/s aggregate useful payload bandwidth`.

This sums all four links and both directions. With balanced symmetric traffic, it is approximately **2.30 GB/s per device bidirectionally**, or **1.15 GB/s per direction per device**. Host CPU reduction work, host memory bandwidth, protocol overhead and fixed collective latency are additional requirements, not specified free resources.

This is the **mean-context** calculation. To maintain at least 70% of that same mean-context ceiling even at the last decode step, the larger 53.849396-ms local step requires approximately **9.95 GB/s**, still assuming zero fixed host latency. If the target instead tracks 70% of the instantaneous context-dependent ceiling, recompute the denominator at each context.

For fixed latency lambda per hidden-state collective and other host time H:

`B_host >= D_host / (T_target - T_local - 64*lambda - H)`.

A nonpositive denominator makes the target impossible regardless of bandwidth. A traffic-only perfect-overlap necessary condition is about **1.99 GB/s**, but it is not sufficient because successive operations depend on completed reductions. If “theoretical maximum” means the zero-network **modeled schedule rate of 18.95**, rather than the ideal streaming ceiling, its separate 70% target needs approximately **5.94 GB/s** in the same serialized model. The definition must remain explicit.

Finally, **the host bandwidth adequate for decode does not imply negligible prefill communication**. Under the same 64 FP32 collectives, each prefill payload is 2048 times larger: total host traffic is approximately **256 GiB**, versus 128 MiB per decode forward. At 9.203 GB/s, serialized prefill traffic alone takes about **29.87 s**. Added to the 46.06-s local estimate, this gives roughly **75.92 s warm four-device TTFT**, before fixed host latency and extra chunking overhead. Prefill/decode communication must therefore be modeled separately even when the tensor-parallel partition is the same.
