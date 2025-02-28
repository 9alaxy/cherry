mkdir ./log

device_number=0

num_batch=(4)
epoch=5

fan_out=10,25,30
layer=3

# reddit ogbn-arxiv ogbn-products amazon ogbn-papers100M
data=(ogbn-products)
hidden=(256)
method=REG

# SAGE GCN GAT
model=SAGE

for da in ${data[@]}
do  
    save_path=./ics_log
    mkdir $save_path
    for hid in ${hidden[@]}
    do
        for nb in ${num_batch[@]}
        do
            save_name=${method}-${nb}-batch-${layer}-layer-${hid}-hid-${model}-${da}-ics.log
            echo $save_name
            python3 Betty.py \
                --dataset $da \
                --aggre mean \
                --seed 1236 \
                --setseed True \
                --GPUmem True \
                --selection-method $method \
                --re-partition-method $method \
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
                --model $model \
                --load-full-batch False \
                > ${save_path}/${save_name}
        done
    done
done

mkdir ./log

device_number=0

num_batch=(4)
epoch=5

fan_out=10,25,30
layer=3

# reddit ogbn-arxiv ogbn-products amazon ogbn-papers100M
data=(ogbn-products)
hidden=(256)
method=Cherry

# SAGE GCN GAT
model=SAGE

for da in ${data[@]}
do  
    save_path=./ics_log
    mkdir $save_path
    for hid in ${hidden[@]}
    do
        for nb in ${num_batch[@]}
        do
            save_name=${method}-${nb}-batch-${layer}-layer-${hid}-hid-${model}-${da}-ics.log
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
                --model $model \
                > ${save_path}/${save_name}
        done
    done
done

aggr=mean
method=vanilla

for da in ${data[@]}
do  
    save_path=./ics_log
    mkdir $save_path
    for hid in ${hidden[@]}
    do
        for md in ${model[@]}
        do
            for nb in ${num_batch[@]}
            do
                save_name=${method}-${nb}-batch-${layer}-layer-${hid}-hid-${md}-${da}-ics.log
                echo $save_name
                python3 vanilla_cherry.py \
                    --dataset $da \
                    --aggre $aggr \
                    --seed 1236 \
                    --setseed True \
                    --GPUmem True \
                    --selection-method $method \
                    --re-partition-method $method \
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
                    --load-full-batch False \
                    > ${save_path}/${save_name}
            done
        done
    done
done