mkdir ./log

device_number=0

num_batch=(8)
epoch=6

fan_out=10,25,30
layer=3

# # reddit ogbn-arxiv ogbn-products amazon ogbn-papers100M
data=(ogbn-arxiv ogbn-products reddit)
hidden=(256)
method=REG

# # SAGE GCN GAT
model=(SAGE GCN)
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
                python3 Betty.py \
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

# for da in ${data[@]}
# do  
#     save_path=./log/${da}
#     mkdir $save_path
#     for hid in ${hidden[@]}
#     do
#         for fan in ${fan_out[@]}
#         do
#             for nb in ${num_batch[@]}
#             do
#                 save_name=${method}-${nb}-batch-${layer}-layer-${hid}-hid-${model}-${da}-${aggr}-${fan}.log
#                 echo $save_name
#                 python3 Betty.py \
#                     --dataset $da \
#                     --aggre $aggr \
#                     --seed 1236 \
#                     --setseed True \
#                     --GPUmem True \
#                     --selection-method $method \
#                     --re-partition-method $method \
#                     --num-batch $nb \
#                     --lr 0.01 \
#                     --num-runs 1 \
#                     --num-epochs $epoch \
#                     --num-layers $layer \
#                     --num-hidden $hid \
#                     --dropout 0.5 \
#                     --fan-out $fan \
#                     --device-number $device_number \
#                     --num-heads 4 \
#                     --model $model \
#                     --load-full-batch False \
#                     > ${save_path}/${save_name}
#             done
#         done
#     done
# done

# num_batch=(2 4 8 16 32)
# fan_out=10,5,5
# layer=3

# hid=32
# # SAGE GCN GAT
# model=(SAGE GCN)
# data=(reddit ogbn-arxiv ogbn-products)

# method=REG

# for da in ${data[@]}
# do  
#     save_path=./log/${da}
#     mkdir $save_path
#     for mdl in ${model[@]}
#     do
#         for nb in ${num_batch[@]}
#         do
#             save_name=${method}-${nb}-batch-${layer}-layer-${hid}-hid-${mdl}-${da}.log
#             echo $save_name
#             python3 Betty.py \
#                 --dataset $da \
#                 --aggre mean \
#                 --seed 1236 \
#                 --setseed True \
#                 --GPUmem True \
#                 --selection-method $method \
#                 --re-partition-method $method \
#                 --num-batch $nb \
#                 --lr 0.01 \
#                 --num-runs 1 \
#                 --num-epochs $epoch \
#                 --num-layers $layer \
#                 --num-hidden $hid \
#                 --dropout 0.5 \
#                 --fan-out $fan_out \
#                 --device-number $device_number \
#                 --num-heads 4 \
#                 --model $mdl \
#                 --load-full-batch False \
#                 > ${save_path}/${save_name}
#         done
#     done
# done

# hid=16
# # SAGE GCN GAT
# model=(GAT)

# for da in ${data[@]}
# do  
#     save_path=./log/${da}
#     mkdir $save_path
#     for mdl in ${model[@]}
#     do
#         for nb in ${num_batch[@]}
#         do
#             save_name=${method}-${nb}-batch-${layer}-layer-${hid}-hid-${mdl}-${da}.log
#             echo $save_name
#             python3 Betty.py \
#                 --dataset $da \
#                 --aggre mean \
#                 --seed 1236 \
#                 --setseed True \
#                 --GPUmem True \
#                 --selection-method $method \
#                 --re-partition-method $method \
#                 --num-batch $nb \
#                 --lr 0.01 \
#                 --num-runs 1 \
#                 --num-epochs $epoch \
#                 --num-layers $layer \
#                 --num-hidden $hid \
#                 --dropout 0.5 \
#                 --fan-out $fan_out \
#                 --device-number $device_number \
#                 --num-heads 4 \
#                 --model $mdl \
#                 --load-full-batch False \
#                 > ${save_path}/${save_name}
#         done
#     done
# done

# fan_out=10,25,30
# hid=128
# # SAGE GCN GAT
# model=(GAT)
# num_batch=(4 8 16)

# for da in ${data[@]}
# do  
#     save_path=./log/${da}
#     mkdir $save_path
#     for mdl in ${model[@]}
#     do
#         for nb in ${num_batch[@]}
#         do
#             save_name=${method}-${nb}-batch-${layer}-layer-${hid}-hid-${mdl}-${da}.log
#             echo $save_name
#             python3 Betty.py \
#                 --dataset $da \
#                 --aggre mean \
#                 --seed 1236 \
#                 --setseed True \
#                 --GPUmem True \
#                 --selection-method $method \
#                 --re-partition-method $method \
#                 --num-batch $nb \
#                 --lr 0.01 \
#                 --num-runs 1 \
#                 --num-epochs $epoch \
#                 --num-layers $layer \
#                 --num-hidden $hid \
#                 --dropout 0.5 \
#                 --fan-out $fan_out \
#                 --device-number $device_number \
#                 --num-heads 4 \
#                 --model $mdl \
#                 --load-full-batch False \
#                 > ${save_path}/${save_name}
#         done
#     done
# done