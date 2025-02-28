mkdir ./ac_log

device_number=0

num_batch=(4)
epoch=200

fan_out=10,25
layer=2

# reddit ogbn-arxiv ogbn-products amazon ogbn-papers100M
data=(reddit)
hidden=(64)
method=Cherry

# SAGE GCN GAT
model=(SAGE GCN)

for da in ${data[@]}
do  
    save_path=./ac_log/${da}
    mkdir $save_path
    for hid in ${hidden[@]}
    do
        for md in ${model[@]}
        do
            for nb in ${num_batch[@]}
            do
                save_name=${method}-${nb}-batch-${layer}-layer-${hid}-hid-${md}-${da}.log
                echo $save_name
                python3 micro_batch_train.py \
                    --dataset $da \
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
                    --num-hidden $hid \
                    --dropout 0.5 \
                    --fan-out $fan_out \
                    --device-number $device_number \
                    --num-heads 4 \
                    --model $md \
                    --eval \
                    > ${save_path}/${save_name}
            done
        done
    done
done