#!/bin/bash

set -e
set -x

# Set up a trap to catch errors and exit all processes
trap 'echo "Error caught, exiting all processes..."; kill 0; exit 1' ERR


CKPT=$1
TASK=$2
SAVE_DIR=$3
CONV_TEMPLATE=$4
MODEL_NAME=$5
MAX_NUM_FRAMES=$6
LLMS_EVAL_MODEL=$7

TASK_SUFFIX="${TASK//,/_}"

PORT=$(python -c "import socket; s = socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

accelerate launch --num_processes 2 --main_process_port $PORT -m lmms_eval \
    --model $LLMS_EVAL_MODEL \
    --model_args pretrained="$CKPT",conv_template=$CONV_TEMPLATE,model_name=$MODEL_NAME,torch_dtype=bfloat16,max_frames_num=$MAX_NUM_FRAMES \
    --tasks $TASK \
    --batch_size 1 \
    --limit 4 \
    --verbosity=DEBUG \
    --log_samples \
    --log_samples_suffix $TASK_SUFFIX \
    --output_path "$SAVE_DIR"
