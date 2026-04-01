set -o pipefail

device_number=1

# Log output directory (configurable)
save_dir=${1:-./exp}

num_batch=(4)
epoch=20

fan_out=5,10,25
layer=3

# reddit ogbn-arxiv ogbn-products amazon ogbn-papers100M
data=(ogbn-products)
hidden=(256)
method=Berry

# SAGE GCN GAT
model=(GCN)
for da in ${data[@]}
do
    save_path=${save_dir}/${da}
    mkdir -p "$save_path"
    for hid in ${hidden[@]}
    do
        for md in ${model[@]}
        do
            nb=2
            while [ $nb -le 64 ]
            do
                save_name=${method}-${nb}-batch-${layer}-layer-${hid}-hid-${md}-${da}-${epoch}.log
                log_path=${save_path}/${save_name}
                echo "$save_name"

                if [ -f "$log_path" ]; then
                    echo "[SKIP] 日志已存在，不会覆盖: $log_path"
                    echo "[提示] 如需重新生成，请先删除或重命名该日志文件。"
                    nb=$((nb * 2))
                    continue
                fi

                # Stream to terminal and save to log simultaneously.
                python3 -u micro_batch_train_berry.py \
                    --dataset $da \
                    --aggre pool \
                    --seed 1236 \
                    --setseed True \
                    --GPUmem True \
                    --selection-method $method \
                    --num-batch $nb \
                    --lr 0.01 \
                    --num-runs 1 \
                    --num-epochs $epoch \
                    --num-layers $layer \
                    --num-hidden $hid \
                    --dropout 0.5 \
                    --fan-out $fan_out \
                    --device-number $device_number \
                    --num-heads 4 \
                    --model $md \
                    --aggre pool \
                    --eval \
                    2>&1 | tee "$log_path"

                nb=$((nb * 2))
            done
        done
    done
done
