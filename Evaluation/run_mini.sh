mkdir ./log

device_number=0

# num_batch=(1 2 4 8 16 32)
epoch=10

fan_out=10,25,30
layer=3

hid=256
# reddit ogbn-arxiv ogbn-products amazon ogbn-papers100M
data=(ogbn-papers100M)
model=SAGE
nb=4
for da in ${data[@]}
do
    save_path=./log/${da}
    mkdir $save_path
    save_name=mini-${nb}-batch-${layer}-layer-${hid}-hid-${model}-${da}.log
    echo $save_name
    python3 mini_batch_train.py \
        --dataset $da \
        --num-batch $nb \
        --num-layers $layer \
        --lr 0.01 \
        --fan-out $fan_out \
        --num-hidden $hid \
        --num-runs 1 \
        --num-epoch $epoch \
        --device-number $device_number \
        --num-heads 4 \
        --model $model \
        --aggre mean \
        --load-full-batch True \
        > ${save_path}/${save_name}
done