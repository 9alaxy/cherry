# Memory-aware Partition Effectiveness Report

## Run Summary

| run_id | model | method | memory_aware | budget_gb | init_batch | final_batch | peak_train_gb | gpu_peak_gb | time_sum_s | final_test_acc | status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| GCN_cherry_warmup | GCN | Cherry | 0 | 0.0000 | 4 | 4 | 0.2353 | 0.9131 | 0.3390 |  | ok |
| GCN_cherry_baseline | GCN | Cherry | 0 | 0.0000 | 4 | 4 | 0.2354 | 0.9131 | 1.5495 |  | ok |
| GCN_metis_baseline | GCN | Metis | 0 | 0.0000 | 4 | 4 | 0.3689 | 1.0146 | 2.4367 |  | ok |
| GCN_berry_relaxed | GCN | Berry | 1 | 0.2165 | 2 | 2 | 0.3315 | 1.0381 | 1.2167 |  | ok |
| GCN_berry_tight | GCN | Berry | 1 | 0.1883 | 2 | 2 | 0.3316 | 1.0381 | 1.1740 |  | ok |

## Criteria Check

| model | mem_drop_tight_vs_cherry_pct | tight_budget_trainable | delta_test_acc_pp | time_ratio_relaxed_vs_cherry | C1(mem>=10%) | C2(budget) | C3(acc) | C4(time) |
|---|---:|---|---:|---:|---|---|---|---|
| GCN | -40.86 | no |  | 0.785 | FAIL | FAIL | N/A | PASS |

## Final Verdict

FAIL
