# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Cherry is a research implementation for efficient large-scale GNN (Graph Neural Network) training that enables training beyond GPU memory limits using micro-batching techniques. The project is built on DGL (Deep Graph Library) with PyTorch backend and targets Intel x86 processors with NVIDIA A100 GPUs.

### Problem Addressed

Cherry solves the "GPU Memory Wall" problem in large-scale GNN training. When graphs become extremely large (billions of nodes and edges), node features, graph structure, and intermediate computation results consume vast GPU memory, often exceeding single GPU capacity.

### Key Innovations

1. **Two Micro-Batching Modes**:
   - **Cherry-GMFG** (Global Memory-oriented Fusion with Graph-aware Partitioning): Uses out-degree centric graph partitioning to split the large graph into partitions stored in CPU memory
   - **Cherry-LMFG** (Local Memory-oriented Fusion with Block-based Loading): Uses block-based data loader for local memory scenarios

2. **Out-degree Centric Graph Partitioning**: Optimizes for computational load balancing rather than just minimizing edge cuts

3. **System-level Optimizations**: Co-design of data loading, memory management, and computation pipeline

## Architecture

### Core Components

1. **Micro-Batch Training System** (`pytorch/micro_batch_train/`)
   - `micro_batch_train.py` - Main training script for Cherry-GMFG
   - `cherry_block_dataloader.py` - Data loader for Cherry-LMFG (Local Memory variant)
   - `c_block_dataloader.py` - Alternative data loader implementation
   - `cherry_graph_partitioner.py` - Implements Out-degree Centric Graph Partitioning
   - `vanilla_cherry.py` - Vanilla baseline implementation

2. **GNN Models** (`pytorch/models/`)
   - `gcn_model_cherry.py` - GCN implementation adapted for Cherry
   - `graphsage_model.py` - GraphSAGE model
   - `deep_gcn_model.py` - DeepGCN variant
   - `gat_model_cherry.py` - GAT (Graph Attention Network) model

3. **Utilities** (`pytorch/utils/`)
   - `load_graph.py` - Dataset loading (OGB datasets, Reddit, Amazon)
   - `memory_usage.py` / `cpu_mem_usage.py` - Memory profiling utilities
   - `utils.py` - Common utilities and Logger
   - `my_utils.py` - Additional helper functions

4. **Evaluation Framework** (`Evaluation/`)
   - Multiple bash scripts for running experiments on different datasets
   - Baseline comparison with vanilla methods and Betty

### Performance Analysis Modules (`Evaluation/profile/`)

The project includes comprehensive performance analysis tools:

| Module | Description |
|--------|-------------|
| `computational_efficiency/` | Training speed and throughput analysis |
| `memory_efficiency/` | GPU memory usage and peak memory analysis |
| `partitioning_quality/` | Graph partitioning quality metrics (edge cut, replication factor) |
| `gpu_utilization/` | Real-time GPU utilization monitoring |
| `load_balance/` | Computational load balance across micro-batches |
| `scalability/` | Scalability analysis with varying batch sizes |

**Experiment records**: `Evaluation/profile/experiment_record.md` contains detailed experimental results and findings.

## Running Experiments

**IMPORTANT: Before running ANY Python code, always check and activate the cherry environment first:**

```bash
# Check current environment
conda info --envs | grep -E "^\*"

# Or check python path
which python
# Should show: /root/miniconda3/envs/cherry/bin/python

# If not cherry, activate it
conda activate cherry
```

### Quick Start

```bash
cd Evaluation/
bash run_GMFG.sh        # Cherry-GMFG (Global Memory variant)
bash rum_LMFG.sh        # Cherry-LMFG (Local Memory variant)
bash run_mini.sh        # Mini-batch baseline
bash run_betty.sh       # Betty baseline comparison
```

### Specialized Analysis Scripts

```bash
# Scalability analysis
bash run_scalability_exp.sh

# GPU utilization monitoring
bash run_gpu_util.sh
```

### Single Experiment Execution

```bash
# Cherry-GMFG example
python3 Evaluation/micro_batch_train.py \
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
python3 Evaluation/vanilla_cherry.py \
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

### Data Collection Scripts

| Script | Purpose |
|--------|---------|
| `micro_batch_train.py` | Main training script |
| `vanilla_cherry.py` | Vanilla baseline |
| `mini_batch_train.py` | Mini-batch baseline |
| `Computation_Load_Balance_collection.py` | Compute load balance analysis |
| `max_memory_collection.py` | Maximum memory usage collection |
| `train_time_collection.py` | Training time collection |
| `Overall_Time_Collection.py` | Overall time analysis |
| `data_collection.py` | General data collection |

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
- `--num-runs`: Number of runs for averaging results

## Experiment Recording

**IMPORTANT: After completing each experiment, always record it:**

- Record location: the relevant analysis subdirectory
- For new analysis directories, create an `experiment_record.md` with:
  - Experiment description
  - Complete Python command used
  - Log file name
  - Key findings and metrics

## Project Structure

```
/workspace/Cherry/
├── .claude/                    # Claude Code configuration
├── Evaluation/                 # Experiment scripts and analysis
│   ├── profile/               # Performance analysis modules
│   │   ├── computational_efficiency/
│   │   ├── memory_efficiency/
│   │   ├── partitioning_quality/
│   │   ├── gpu_utilization/
│   │   ├── load_balance/
│   │   ├── scalability/
│   │   ├── SOP.md             # Standard operating procedures
│   │   └── experiment_record.md
│   ├── log/                   # Detailed experiment logs 暂时废弃，禁止修改
│   ├── ac_log/                # Additional runtime logs  暂时废弃，禁止修改
│   └── *.sh                   # Various experiment scripts
├── pytorch/                   # Core implementation
│   ├── micro_batch_train/     # Micro-batch training system
│   ├── models/               # GNN model implementations
│   └── utils/                # Utility functions
├── test/                      # Test scripts
├── cherry.md                  # Project description and innovation points
├── environment.yml            # Conda environment configuration
└── requirements.sh            # Additional requirements
```

## Testing

```bash
# Run performance analysis tests
python3 test/test_performance_analysis.py
```
