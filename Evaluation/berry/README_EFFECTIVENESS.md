# Memory-aware Effectiveness Quick Experiment

This quick pipeline is designed to validate whether memory-aware partition is useful with the following claims:

1. Lower peak memory than non-memory-aware baselines.
2. Under a tight memory budget, memory-aware partition remains trainable by adapting micro-batch count.

## Scripts

- `run_memory_aware_effectiveness.sh`
  - Orchestrates warmup, baseline runs, memory-aware runs, and summary generation.
- `summarize_memory_aware_results.py`
  - Parses logs and exports detailed metrics + criteria checks.
- `run_berry_train.sh`
  - Unified training launcher for Cherry / Metis / Berry with optional memory-aware partition.
- `run_prefetch_ablation.sh`
  - One-click baseline (`ap0`) vs async prefetch (`ap1`) ablation on Berry.
- `summarize_prefetch_results.py`
  - Parses matched baseline/async logs and exports CSV + Markdown summary.

## Default Setup (Quick)

- Dataset: `ogbn-arxiv`
- Models: `GCN GAT`
- Device: `DEVICE_NUMBER=1` (RTX 3090)
- Baselines: `Cherry`, `Metis`
- Memory-aware: `Berry`
- Epochs: `10`
- Warmup epochs: `1`
- Baseline init micro-batch: `4`
- Berry init micro-batch: `2`

## Run

```bash
cd /workspace/Cherry/Evaluation/berry
./run_memory_aware_effectiveness.sh
```

## Optional Environment Variables

```bash
DATASET=ogbn-arxiv
MODELS="GCN GAT"
DEVICE_NUMBER=1
NUM_EPOCHS=10
WARMUP_EPOCHS=1
BASELINE_NUM_BATCH=4
BERRY_INIT_BATCH=2
NUM_HIDDEN=64
NUM_LAYERS=2
FAN_OUT=10,25
RELAXED_RATIO=0.92
TIGHT_RATIO=0.80
```

If you already have a fitted beta model, pass it to the launcher:

```bash
BETA_PATH=/abs/path/to/beta.json ./run_memory_aware_effectiveness.sh
```

## Prefetch Ablation (Methodology)

This section is the recommended and reproducible process to compare Berry with/without async prefetch.

### Goal

Under exactly the same setup, compare:

1. `ENABLE_ASYNC_PREFETCH=0` (baseline)
2. `ENABLE_ASYNC_PREFETCH=1` (async prefetch)

and report:

1. `total_time`
2. `Max Memory Allocated` (from main log)
3. `nvidia-smi` peak memory (from GPU sampling log)

### Command (One-click)

```bash
cd /workspace/Cherry/Evaluation/berry
chmod +x run_prefetch_ablation.sh
DATASET=ogbn-arxiv MODEL=GCN NUM_BATCH=4 NUM_LAYERS=2 NUM_HIDDEN=64 FAN_OUT=10,25 DEVICE_NUMBER=1 ./run_prefetch_ablation.sh
```

Example for products:

```bash
DATASET=ogbn-products MODEL=GCN NUM_BATCH=4 NUM_LAYERS=2 NUM_HIDDEN=64 FAN_OUT=10,25 DEVICE_NUMBER=1 ./run_prefetch_ablation.sh
```

### Important Constraints

1. Use `DEVICE_NUMBER=1` in this environment (RTX 3090).
2. Keep all non-prefetch hyper-parameters identical between the two runs.
3. Keep both logs for each run:
  - main log (`train_*.log`)
  - GPU sampling log (`train_*_gpu_mem.log`)

### Produced Artifacts

The one-click script outputs:

1. Paired run logs under `Evaluation/berry/logs/`
2. Auto summary CSV:
  - `Evaluation/berry/logs/prefetch_summary_<dataset>_<model>_<timestamp>.csv`
3. Auto summary Markdown:
  - `Evaluation/berry/logs/prefetch_summary_<dataset>_<model>_<timestamp>.md`

### If You Already Have Logs

You can summarize directly without rerunning:

```bash
python3 summarize_prefetch_results.py \
  --dataset ogbn-arxiv \
  --model GCN \
  --aggre mean \
  --num-batch 4 \
  --num-layers 2 \
  --num-hidden 64 \
  --memory-aware 0
```

## Outputs

Each run writes two logs:

- Main stdout/stderr log: `Evaluation/berry/logs/train_*.log`
- GPU memory sampling log: `Evaluation/berry/logs/train_*_gpu_mem.log`

Experiment-level artifacts are grouped under:

- `Evaluation/berry/logs/effectiveness_<dataset>_<timestamp>/manifest.csv`
- `Evaluation/berry/logs/effectiveness_<dataset>_<timestamp>/summary.csv`
- `Evaluation/berry/logs/effectiveness_<dataset>_<timestamp>/summary_model_summary.csv`
- `Evaluation/berry/logs/effectiveness_<dataset>_<timestamp>/summary.md`

## Criteria Used in Summary

- C1: Tight-budget memory drop of Berry vs Cherry >= 10%
- C2: Tight-budget trainability (baseline over budget while Berry within budget)
- C3: Test accuracy drop <= 1 percentage point (Berry relaxed vs Cherry baseline)
- C4: Time ratio <= 1.25 (Berry relaxed vs Cherry baseline)

These criteria are configurable in code if needed.
