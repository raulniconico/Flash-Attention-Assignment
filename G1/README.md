# Section 2.2 submission

Start with `../../../Téléchargements/section_2_2_submission/section_2_2_report.md`. It answers all five requested items and explains which results are conditional on missing hardware details.

## Reproduce

Requires Python 3.9 or later and only the standard library. From this directory:

```bash
python3 reference_attention.py
python3 performance_model.py --output results.json
```

The numerical reference reports 19 attention comparisons plus split-KV/empty-state checks. The performance program prints and saves the analytical results, including capacities, prefill/decode timing, four-way host traffic and bandwidth, sensitivity, and proposed changes. It asserts that the modeled dense SRAM allocations fit 16 MiB.

## Files

- `../../../Téléchargements/section_2_2_submission/section_2_2_report.md`: complete technical report.
- `kernel_pseudocode.txt`: controller orchestration and vector compute programs; optional array mapping.
- `performance_model.py`: conditional analytical timing calculator.
- `results.json`: generated numeric results used by the report.
- `reference_attention.py`: small independent attention reference and tests.
- `README.md`: these instructions.

## Interpretation

The primary implementation takes the assignment's memory-only vector CPU literally. It assumes at least 12 KiB vector-local working storage, vector support for the chosen 16-bit activation representation, FP32 running state, W8 per-channel scales, preloaded weights, and greedy sampling. It does not silently requantize A16 to A8. The report separately discusses how to use the arrays and the missing numerical/timing contracts.

These are not measured hardware results. The test validates attention indexing and the online-softmax recurrence using Python floating point; it does not validate device execution, low-precision rounding, model quality, queue timings, or host collectives. No actual model weights are included or needed.

The source attachments are not included in this bundle. The scripts run locally without network access and do not send data anywhere.
