#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

task='vg' # seg, cd, cls, ris, vg, od

config=configs/${task}.yaml
save_path=exps/${task}/Mini-RefDIOR_VG/convnext_tiny_lpva

mkdir -p $save_path

CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.run \
    --nproc_per_node=$1 \
    --master_addr=localhost \
    --master_port=$2 \
    train.py \
    --config=$config \
    --save-path $save_path --port $2 \
    ${@:3} 2>&1 | tee $save_path/$now.log
