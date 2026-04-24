#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

config=configs/dinov3_ct_ccformer.yaml
save_path=exps_dinov3/RefDIOR_RIS_Ablation/convnext_tiny_ccformer_sup1

mkdir -p $save_path

CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.run \
    --nproc_per_node=$1 \
    --master_addr=localhost \
    --master_port=$2 \
    train.py \
    --config=$config \
    --save-path $save_path --port $2 \
    ${@:3} 2>&1 | tee $save_path/$now.log
