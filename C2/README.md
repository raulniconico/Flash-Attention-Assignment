# AI inference accelerator — flash-attention kernel and Llama 3.1 8B performance study

Answer to section 2.2 of the technical assignment (problems 1–5).

## Files

| File | What it is |
|---|---|
| `report.md` | The report: step-by-step methodology (§0.1), architecture reading, kernel design, results for tasks 1–5, assumptions A1–A15 (§1.3), decision log with rejected alternatives (§3.7), limitations. **Start here.** |
| `flash_attention_pseudocode.py` | Problem 1: control-CPU pseudocode. Part A = prefill kernel (compute-bound, both systolic arrays), Part B = decode kernel (memory-bound, DMA stream + vector CPU). Runtime API, numerics, tiling, SRAM maps and schedules in the header. Not executable by design. |
| `perf_model.py` | Problems 2–5: analytical performance model built only from the constants of the assignment table. Produces every number quoted in the report (tile timing, elasticities, TTFT, interactivity, improvement variants, 4-accelerator analysis). |
| `model_output.txt` | Console log of `perf_model.py` — the numbers as generated. |
| `results.json` | Same results in machine-readable form. |
| `roofline.png` | Figure 2 of the report: roofline derived from Figure 1 (compute peak vs DRAM bandwidth) with the four workload phases placed on it. |

## Reproducing the numbers

Requirements: Python ≥ 3.9, standard library only. `matplotlib` is needed only for `--plot`.

```bash
python3 perf_model.py                        # prints sections A–G (see model_output.txt)
python3 perf_model.py --json results.json    # also dump raw numbers
python3 perf_model.py --plot                 # also regenerate roofline.png
```

The model is parametric: change any field of `Arch`, `Workload` or `LlamaModel` at
the top of `perf_model.py` (or use `dataclasses.replace` as the improvement and
multi-accelerator sections do) and rerun. Runtime is a few seconds.

## Headline results

| | |
|---|---|
| Peak systolic throughput | 1 536 MAC/cycle (3.07 TOPS INT8); DRAM→SRAM 64 GB/s; ridge 24 MAC/B |
| Flash-attention kernel | prefill: 1 365 MAC/cycle sustained (89 % of peak), 226 ms/layer; decode: DRAM-bound KV stream on the vector CPU, 2.2 ms/layer |
| Bottleneck (problem 2) | prefill kernel: SRAM↔array port bandwidth, elasticity −1.0 (output ports −0.64); decode kernel: DRAM→DMA→SRAM path, −1.0 |
| TTFT (problem 3, prefill) | ≈ 157 s for the 16 × 2048-token batch (compute-bound) |
| Interactivity (problem 3, decode) | ≈ 5.3 tok/s per sequence, 84.5 tok/s aggregate (189 ms/step = 117 ms weights + 71 ms KV, DRAM-bound) |
| Improvements (problem 4) | ×4 memory bandwidth → 8.7 tok/s; INT4 weights + INT8 KV via DMA dequant → 8.7 tok/s; both → 10.3 tok/s |
| 4 accelerators (problem 5) | prefill data-parallel (39 s, no communication), decode tensor-parallel: max 21 tok/s per sequence; ≥ 0.82 GB/s per accelerator link (3.3 GB/s aggregate) for 70 % |

## Key assumptions (full list with rationale in `report.md`, Appendix A)

- **A2** Systolic-array tile time = max(input bytes / input port, output bytes / output port), pipelined; the arrays are port-bound, not PE-bound. (Alternative doubles compute-bound times.)
- **A3** Operands are dynamically re-quantised to INT8 at the array boundary (per-row / per-block scales); 16-bit is kept for storage, residuals, softmax and KV cache. (Exact hi/lo-byte alternative doubles TTFT.)
- **A4** Arrays cannot accumulate across tile ops → FP32 partial-sum accumulation on the vector CPU.
- **A7** Control-CPU commands are queued; the 200/300/100-cycle latencies are pipeline-fill costs. (Without queueing the arrays lose 44 % of peak.)
- **A10** Decode attention runs on the vector CPU, not the arrays.
- **A12** Accelerators communicate only through the host; all-reduces are host-mediated and on the critical path.
- **A13** Within a phase the DMA, arrays and vector CPU overlap perfectly (double buffering); phases are serialised.
