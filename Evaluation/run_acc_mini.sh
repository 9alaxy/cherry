mkdir ./ac_log

device_number=0

# num_batch=(1 2 4 8 16 32)
epoch=500

fan_out=10,25
layer=2

hid=64
# reddit ogbn-arxiv ogbn-products amazon ogbn-papers100M
data=(ogbn-arxiv)
model=(SAGE GCN)
nb=1
for da in ${data[@]}
do
    save_path=./ac_log/${da}
    mkdir $save_path
    for md in ${model[@]}
    do
        save_name=mini-${nb}-batch-${layer}-layer-${hid}-hid-${md}-${da}.log
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
            --model $md \
            --aggre mean \
            --load-full-batch False \
            --eval \
            > ${save_path}/${save_name}
    done
done

