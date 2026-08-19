#!/bin/bash
image_input_dir=$1
image_output_dir=$2
job_id=$3
model_file=$4

GPU_NUM=$(nvidia-smi --list-gpus | wc -l)
$(which python) -m torch.distributed.launch \
    --nproc_per_node=$GPU_NUM \
    --master_port=50016 \
    --use_env \
    run.py run --image_input_dir=${image_input_dir} --image_output_dir=${image_output_dir} --job_id=$job_id --model_file=${model_file}
