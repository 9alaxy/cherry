mkdir ./log

device_number=0

num_batch=(8)
epoch=6

fan_out=10,25,30
layer=3

# reddit ogbn-arxiv ogbn-products amazon ogbn-papers100M
data=(ogbn-products)
hidden=(256)
method=vanilla

# # SAGE GCN GAT
model=(GCN)
aggr=mean

for da in ${data[@]}
do  
    save_path=./log/${da}
    mkdir $save_path
    for hid in ${hidden[@]}
    do
        for md in ${model[@]}
        do
            for nb in ${num_batch[@]}
            do
                save_name=${method}-${nb}-batch-${layer}-layer-${hid}-hid-${md}-${da}.log
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

data=(reddit ogbn-arxiv ogbn-products)
hidden=(128)
model=(GAT)

for da in ${data[@]}
do  
    save_path=./log/${da}
    mkdir $save_path
    for hid in ${hidden[@]}
    do
        for md in ${model[@]}
        do
            for nb in ${num_batch[@]}
            do
                save_name=${method}-${nb}-batch-${layer}-layer-${hid}-hid-${md}-${da}.log
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