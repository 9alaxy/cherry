# Berry Prefetch Ablation Summary

- dataset: ogbn-products
- model: GCN
- aggre: mean
- num_batch: 4
- num_layers: 2
- num_hidden: 64
- memory_aware: 0

## Metrics

- total_time baseline: 1.435017 s
- total_time async: 1.027849 s
- total_time improvement: 28.37%
- torch peak baseline: 1.408388 GB
- torch peak async: 1.201610 GB
- torch peak reduction: 14.68%
- nvidia-smi peak baseline: 2529 MiB
- nvidia-smi peak async: 2205 MiB
- nvidia-smi peak reduction: 12.81%
- async prefetch_wait_time: 0.383212 s

## Logs

- baseline main: ./logs/train_berry_ogbn-products_GCN_mean_b4_l2_h64_ma0_ap0_20260329_060953.log
- async main: ./logs/train_berry_ogbn-products_GCN_mean_b4_l2_h64_ma0_ap1_20260329_061132.log
- baseline gpu: ./logs/train_berry_ogbn-products_GCN_mean_b4_l2_h64_ma0_ap0_20260329_060953_gpu_mem.log
- async gpu: ./logs/train_berry_ogbn-products_GCN_mean_b4_l2_h64_ma0_ap1_20260329_061132_gpu_mem.log
