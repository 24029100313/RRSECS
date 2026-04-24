#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

#task='recs' # seg, cd, cls, ris, vg, od, recs
#
#config=configs/${task}.yaml
#save_path=exps/${task}/RefDIOR_RECS/swin_tiny_ccformer_v3.1
#
#mkdir -p $save_path
#
#python -m torch.distributed.run \
#    --nproc_per_node=$1 \
#    --master_addr=localhost \
#    --master_port=$2 \
#    train.py \
#    --config=$config \
#    --save-path $save_path --port $2 \
#    ${@:3} 2>&1 | tee $save_path/$now.log



config=configs/dinov3_ct_ccformer.yaml
save_path=exps_dinov3/RefDIOR_RIS_Ablation/convnext_tiny_ccformer_sup10

mkdir -p $save_path

python -m torch.distributed.run \
    --nproc_per_node=$1 \
    --master_addr=localhost \
    --master_port=$2 \
    train.py \
    --config=$config \
    --save-path $save_path --port $2 \
    ${@:3} 2>&1 | tee $save_path/$now.log
