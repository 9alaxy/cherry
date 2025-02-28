mkdir ./log

device_number=1

num_batch=(8)
epoch=10

fan_out=10,25,30
layer=3

hidden=(128)
# SAGE GCN GAT
model=(GAT)
# Cherry Metis Random
method=(Random)
# reddit ogbn-arxiv ogbn-products amazon ogbn-papers100M
data=(ogbn-products)
aggr=mean

for da in ${data[@]}
do  
    save_path=./log/${da}
    mkdir $save_path
    for mtd in ${method[@]}
    do
        for mdl in ${model[@]}
        do
            for hid in ${hidden[@]}
            do
                for nb in ${num_batch[@]}
                do
                    save_name=${mtd}-${nb}-batch-${layer}-layer-${hid}-hid-${mdl}-${da}.log
                    echo $save_name
                    python3 micro_batch_train.py \
                        --dataset $da \
                        --aggre $aggr \
                        --seed 1236 \
                        --setseed True \
                        --GPUmem True \
                        --selection-method $mtd \
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
                        --model $mdl \
                        > ${save_path}/${save_name}
                done
            done
        done
    done
done

# for da in ${data[@]}
# do  
#     save_path=./log/${da}
#     mkdir $save_path
#     for mtd in ${method[@]}
#     do
#         for mdl in ${model[@]}
#         do
#             for hid in ${hidden[@]}
#             do
#                 for nb in ${num_batch[@]}
#                 do
#                     save_name=${mtd}-${nb}-batch-${layer}-layer-${hid}-hid-${mdl}-${da}-${aggr}.log
#                     echo $save_name
#                     python3 micro_batch_train.py \
#                         --dataset $da \
#                         --aggre $aggr \
#                         --seed 1236 \
#                         --setseed True \
#                         --GPUmem True \
#                         --selection-method $mtd \
#                         --num-batch $nb \
#                         --lr 0.01 \
#                         --num-runs 1 \
#                         --num-epochs $epoch \
#                         --num-layers $layer \
#                         --num-hidden $hid \
#                         --dropout 0.5 \
#                         --fan-out $fan_out \
#                         --device-number $device_number \
#                         --num-heads 4 \
#                         --model $mdl \
#                         > ${save_path}/${save_name}
#                 done
#             done
#         done
#     done
# done

# hid=128
# SAGE GCN GAT
# model=(GAT)

# for da in ${data[@]}
# do  
#     save_path=./log/${da}
#     mkdir $save_path
#     for mtd in ${method[@]}
#     do
#         for mdl in ${model[@]}
#         do
#             for nb in ${num_batch[@]}
#             do
#                 save_name=${mtd}-${nb}-batch-${layer}-layer-${hid}-hid-${mdl}-${da}.log
#                 echo $save_name
#                 python3 micro_batch_train.py \
#                     --dataset $da \
#                     --aggre mean \
#                     --seed 1236 \
#                     --setseed True \
#                     --GPUmem True \
#                     --selection-method $mtd \
#                     --num-batch $nb \
#                     --lr 0.01 \
#                     --num-runs 1 \
#                     --num-epochs $epoch \
#                     --num-layers $layer \
#                     --num-hidden $hid \
#                     --dropout 0.5 \
#                     --fan-out $fan_out \
#                     --device-number $device_number \
#                     --num-heads 4 \
#                     --model $mdl \
#                     > ${save_path}/${save_name}
#             done
#         done
#     done
# done

# fan_out=10,25,30
# hid=128
# # SAGE GCN GAT
# model=(GAT)
# data=(reddit ogbn-arxiv ogbn-products)
# num_batch=(4 8 16)

# for da in ${data[@]}
# do  
#     save_path=./log/${da}
#     mkdir $save_path
#     for mtd in ${method[@]}
#     do
#         for mdl in ${model[@]}
#         do
#             for nb in ${num_batch[@]}
#             do
#                 save_name=${mtd}-${nb}-batch-${layer}-layer-${hid}-hid-${mdl}-${da}.log
#                 echo $save_name
#                 python3 micro_batch_train.py \
#                     --dataset $da \
#                     --aggre mean \
#                     --seed 1236 \
#                     --setseed True \
#                     --GPUmem True \
#                     --selection-method $mtd \
#                     --num-batch $nb \
#                     --lr 0.01 \
#                     --num-runs 1 \
#                     --num-epochs $epoch \
#                     --num-layers $layer \
#                     --num-hidden $hid \
#                     --dropout 0.5 \
#                     --fan-out $fan_out \
#                     --device-number $device_number \
#                     --num-heads 4 \
#                     --model $mdl \
#                     > ${save_path}/${save_name}
#             done
#         done
#     done
# done
