#!/bin/bash

CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.run \
    --nproc_per_node=$1 \
    --master_addr=localhost \
    --master_port=$2 \
    eval.py \
    --port $2 \
    ${@:3}
