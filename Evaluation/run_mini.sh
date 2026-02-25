device_number=0

# num_batch=(1 2 4 8 16 32)
epoch=5

fan_out=10,25
#10,25,30,40
layer=2

hid=256
# reddit ogbn-arxiv ogbn-products amazon ogbn-papers100M
data=(ogbn-products)
model=GCN
agg=lstm
nb=1
for da in ${data[@]}
do
    save_path=./log/${model}/${da}
    mkdir -p $save_path
    save_name=mini-${nb}-batch-${layer}-layer-${hid}-hid-${model}-${da}-${agg}-${fan_out}.log
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
        --aggre $agg \
        --load-full-batch True \
        > ${save_path}/${save_name}
done