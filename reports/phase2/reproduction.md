# Phase 2 · Step 1 — Reproduction of Phase-1 results

Recomputed from `models/best_attention_model.pt` + `models/preprocessor.pkl` + processed data; the stored `reports/final_metrics.json` is only used for comparison.

| metric | claimed (README) | stored report | reproduced now | status |
|---|---|---|---|---|
| precision | 0.3180 | 0.3184 | 0.3184 | MATCH |
| recall | 0.4880 | 0.4875 | 0.4875 | MATCH |
| f1 | 0.3850 | 0.3852 | 0.3852 | MATCH |
| roc_auc | 0.8450 | 0.8453 | 0.8453 | MATCH |
| pr_auc | 0.3260 | 0.3257 | 0.3257 | MATCH |
| false_alarms_per_day | 0.1200 | 0.1152 | 0.1152 | MATCH |
| lead_time_median_min | 120.0000 | 120.0000 | 120.0000 | MATCH |
| params | 31425 | — | 31425 | MATCH |
| size_kb | 127 | — | 126.5645 | MATCH |
| torch_latency_ms | 0.4200 | — | 0.4027 | timing (re-measured) |
| onnx_latency_ms | 0.0700 | — | 0.0693 | timing (re-measured) |
| stream_ms | 8.4000 | — | 8.3683 | timing (re-measured) |

Timing values are hardware/load dependent and were re-measured (median of ≥300 calls, 1 thread); they are reported, not compared for equality.

Validation (reproduced): PR-AUC 0.5531, event recall (detectable) 0.889, FA/day 0.082.