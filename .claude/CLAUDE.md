# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Cherry is a research implementation for efficient large-scale GNN (Graph Neural Network) training that enables training beyond GPU memory limits using micro-batching techniques. The project is built on DGL (Deep Graph Library) with PyTorch backend and targets Intel x86 processors with NVIDIA A100 GPUs.

## Architecture

### Core Components

1. **Micro-Batch Training System** (`pytorch/micro_batch_train/`)
   - `micro_batch_train.py` - Main training script for Cherry-GMFG (Graph Micro-Batch Fusion with Global Memory)
   - `cherry_block_dataloader.py` - Data loader for Cherry-LMFG (Local Memory variant)
   - `c_block_dataloader.py` - Alternative data loader implementation
   - `cherry_graph_partitioner.py` - Implements Out-degree Centric Graph Partitioning

2. **GNN Models** (`pytorch/models/`)
   - `gcn_model_cherry.py` - GCN implementation adapted for Cherry
   - `graphsage_model.py` - GraphSAGE model
   - `deep_gcn_model.py` - DeepGCN variant
   - Includes GAT model (referenced but file path may vary)

3. **Utilities** (`pytorch/utils/`)
   - `load_graph.py` - Dataset loading (OGB datasets, Reddit, Amazon)
   - `memory_usage.py` / `cpu_mem_usage.py` - Memory profiling utilities
   - `utils.py` - Common utilities and Logger
   - `my_utils.py` - Additional helper functions

4. **Evaluation Framework** (`Evaluation/`)
   - Multiple bash scripts for running experiments on different datasets
   - Baseline comparison with vanilla methods and Betty (another memory optimization technique)

### Key Algorithms

1. **Cherry-GMFG**: Global memory micro-batching approach with graph partitioning
2. **Cherry-LMFG**: Local memory variant using block-based data loading
3. **Graph Partitioning**: Out-degree centric partitioning for optimal memory distribution

## Running Experiments

**IMPORTANT: Before running ANY Python code, always check and activate the cherry environment first:**

```bash
# Check current environment - if not cherry, activate it
conda activate cherry
```

**Then run experiments:**
```bash
cd Evaluation/
bash run_GMFG.sh
```

**Cherry-LMFG (Local Memory variant):**
```bash
cd Evaluation/
bash rum_LMFG.sh
```

**Vanilla baselines:**
```bash
cd Evaluation/
bash run_vanilla.sh
```

**Evaluation scripts:**
- `run_acc_mini.sh` - Mini-batch accuracy evaluation
- `run_acc_micro.sh` - Micro-batch accuracy evaluation
- `run_time.sh` - Time performance evaluation

### Single Experiment Execution

```bash
# Cherry-GMFG example
python3 pytorch/micro_batch_train/micro_batch_train.py \
    --dataset ogbn-products \
    --model GCN \
    --selection-method Cherry \
    --num-batch 8 \
    --num-layers 3 \
    --num-hidden 256 \
    --num-epochs 10 \
    --fan-out 10,25,30 \
    --device-number 0

# Vanilla Cherry (baseline)
python3 pytorch/micro_batch_train/vanilla_cherry.py \
    --dataset ogbn-products \
    --model GCN \
    --selection-method vanilla \
    --num-batch 8 \
    --num-layers 3 \
    --num-hidden 256 \
    --num-epochs 10 \
    --fan-out 10,25,30 \
    --device-number 0
```

## Dataset Configuration

Edit `pytorch/utils/load_graph.py` to configure dataset storage paths. The project supports:
- Open Graph Benchmark (OGB) datasets (auto-download)
- Reddit dataset
- Amazon dataset (manual download required from GraphSAINT)

## Key Parameters

- `--num-batch`: Number of micro-batches
- `--selection-method`: Cherry (proposed) / vanilla / Metis / Random
- `--re-partition-method`: Graph re-partitioning strategy
- `--fan-out`: Neighborhood sampling sizes for GNN layers
- `--GPUmem`: Enable GPU memory optimization
- `--load-full-batch`: Whether to load full graph batches

## Experiment Recording

**IMPORTANT: After completing each experiment, always record it:**

- Record location: Same directory as the generated `.log` file (e.g., `Evaluation/profile/partitioning_quality/`)
- Record format: Create an `experiment_record.md` file with:
  - Experiment description
  - Complete Python command used
  - Log file name

Example:
```markdown
### Cherry-GMFG (4 micro-batches, 2 layers)
```bash
python3 micro_batch_train.py \
    --dataset ogbn-arxiv \
    --model GCN \
    --selection-method Cherry \
    --num-batch 4 ...
```
- 日志: `cherry_gmfg_test.log`
```

## Project Structure Notes

- The `Evaluation/` directory contains experiment scripts and logs
- Results are saved in `Evaluation/log/` with detailed metrics
- `Evaluation/ac_log/` contains additional runtime logs
- `Evaluation/profile/` contains performance analysis data
- Cherry's code is primarily in `pytorch/micro_batch_train/`
- Betty's implementation is included in `pytorch/micro_batch_train/Betty_file/` for comparison