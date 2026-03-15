#!/bin/bash
# Scalability Experiment: Run Cherry with different num_batch values

cd /workspace/Cherry/Evaluation

mkdir -p ./profile/scalability

device_number=0
num_batches=(2 4 8 16)
epoch=2  # Quick experiment for scalability analysis

fan_out=10,25
layer=2
data=ogbn-arxiv
hidden=64
method=Cherry
model=GCN

echo "========================================"
echo "Scalability Experiment: Cherry"
echo "========================================"

for nb in ${num_batches[@]}
do
    save_name=${method}_${nb}batch_${layer}layer_${hidden}hid_${model}_${data}.log
    echo "Running $method with num_batch=$nb..."

    python3 micro_batch_train.py \
        --dataset $data \
        --aggre mean \
        --seed 1236 \
        --setseed True \
        --GPUmem True \
        --selection-method $method \
        --num-batch $nb \
        --lr 0.01 \
        --num-runs 1 \
        --num-epochs $epoch \
        --num-layers $layer \
        --num-hidden $hidden \
        --dropout 0.5 \
        --fan-out $fan_out \
        --device-number $device_number \
        --num-heads 4 \
        --model $model \
        --eval \
        2>&1 | tee ./profile/scalability/${save_name}

    echo "Completed num_batch=$nb"
    echo "---"
done

echo "========================================"
echo "All experiments completed!"
echo "========================================"
