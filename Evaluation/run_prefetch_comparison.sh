#!/bin/bash
# Prefetch 性能对比实验脚本

LOG_DIR="./ac_log_comparison"
mkdir -p $LOG_DIR

PYTHON="/root/miniconda3/envs/cherry/bin/python3"
SCRIPT="micro_batch_train.py"

# 实验配置
# 格式: layers fanout num_batch
configs=(
    "2 10,25 2"
    "2 10,25 4"
    "2 10,25 8"
    "3 10,25,25 2"
    "3 10,25,25 4"
    "3 10,25,25 8"
)

run_exp() {
    local layers=$1
    local fanout=$2
    local nb=$3
    local use_prefetch=$4

    local prefetch_str=""
    if [ "$use_prefetch" = "True" ]; then
        prefetch_str="prefetch"
    else
        prefetch_str="norm"
    fi

    local log_file="$LOG_DIR/layers${layers}_fanout${fanout}_batch${nb}_${prefetch_str}.log"

    echo "Running: layers=$layers fanout=$fanout batch=$nb prefetch=$use_prefetch"
    echo "Log: $log_file"

    $PYTHON $SCRIPT \
        --dataset  ogbn-products  \
        --aggre mean \
        --seed 1236 \
        --setseed True \
        --GPUmem True \
        --selection-method Cherry \
        --num-batch $nb \
        --lr 0.01 \
        --num-runs 1 \
        --num-epochs 10 \
        --num-layers $layers \
        --num-hidden 64 \
        --dropout 0.5 \
        --fan-out $fanout \
        --device-number 0 \
        --num-heads 4 \
        --model GAT \
        --eval \
        --use-prefetch $use_prefetch \
        > $log_file 2>&1

    echo "Done: $log_file"
    echo "---"
}

# 运行所有实验
for config in "${configs[@]}"; do
    read -r layers fanout nb <<< "$config"

    # 运行 norm 版本
    run_exp $layers $fanout $nb "False"

    # 运行 prefetch 版本
    run_exp $layers $fanout $nb "True"
done

echo "All experiments completed!"
echo "Logs saved to: $LOG_DIR"
