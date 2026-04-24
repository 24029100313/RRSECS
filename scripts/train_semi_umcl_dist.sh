#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")


config=configs/dinov3_ct_ccformer.yaml
save_path=exps_dinov3/RefDIOR_RIS_Ablation/convnext_tiny_ccformer_semi10_umcl_p8

mkdir -p $save_path

python -m torch.distributed.run \
    --nproc_per_node=$1 \
    --master_addr=localhost \
    --master_port=$2 \
    train_semi_umcl.py \
    --config=$config \
    --save-path $save_path --port $2 \
    ${@:3} 2>&1 | tee $save_path/$now.log
