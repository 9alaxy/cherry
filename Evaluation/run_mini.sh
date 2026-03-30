device_number=0

# num_batch=(1 2 4 8 16 32)
epoch=5

fan_out=5,5
#10,25,30,40
layer=2

hid=256
# reddit ogbn-arxiv ogbn-products amazon ogbn-papers100M
data=(ogbn-products)
model=GAT
agg=lstm
nb=1
head_num=3
for da in ${data[@]}
do
    save_path=./mem/num_head/GAT
    mkdir -p $save_path
    save_name=mini_${nb}batch_${layer}layer_${hidden}hid_${model}_${da}_${head_num}head.log
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
        --num-heads $head_num \
        --model $model \
        --aggre $agg \
        --load-full-batch True \
        > ${save_path}/${save_name}
done