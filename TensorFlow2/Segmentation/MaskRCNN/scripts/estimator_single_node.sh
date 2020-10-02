#!/usr/bin/env bash
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

BATCH_SIZE=4
HOST_COUNT=1
GPU_COUNT=`nvidia-smi --query-gpu=name --format=csv,noheader | wc -l`
IMAGES=118287
GLOBAL_BATCH_SIZE=$((BATCH_SIZE * HOST_COUNT * GPU_COUNT))
STEP_PER_EPOCH=$(( IMAGES / GLOBAL_BATCH_SIZE ))
FIRST_DECAY=$(( 8 * STEP_PER_EPOCH ))
SECOND_DECAY=$(( 11 * STEP_PER_EPOCH ))
TOTAL_STEPS=$(( 13 * STEP_PER_EPOCH ))
LR_MULTIPLIER=0.001
BASE_LR=$(echo $GLOBAL_BATCH_SIZE*$LR_MULTIPLIER | bc)

source activate mask_rcnn

BASEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
rm -rf $BASEDIR/../baseline_1x
mkdir -p $BASEDIR/../baseline_1x
/opt/amazon/openmpi/bin/mpirun --allow-run-as-root --tag-output --mca plm_rsh_no_tree_spawn 1 \
    --mca btl_tcp_if_exclude lo,docker0 \
    -np $GPU_COUNT -H localhost:$GPU_COUNT \
    -x NCCL_DEBUG=VERSION \
    -x LD_LIBRARY_PATH \
    -x PATH \
    --oversubscribe \
    /home/ubuntu/anaconda3/envs/mask_rcnn/bin/python ${BASEDIR}/../mask_rcnn_main.py \
        --mode="train_and_eval" \
        --checkpoint="/home/ubuntu/DeepLearningExamples/TensorFlow2/Segmentation/MaskRCNN/resnet/resnet-nhwc-2018-02-07/model.ckpt-112603" \
        --eval_samples=5000 \
        --log_interval=100 \
        --init_learning_rate=$BASE_LR \
        --learning_rate_steps="$FIRST_DECAY,$SECOND_DECAY" \
        --optimizer_type="SGD" \
        --lr_schedule="piecewise" \
        --model_dir="$BASEDIR/../baseline_1x" \
        --num_steps_per_eval=$STEP_PER_EPOCH \
        --warmup_learning_rate=0.000133 \
        --warmup_steps=1000 \
        --global_gradient_clip_ratio=0.0 \
        --total_steps=$TOTAL_STEPS \
        --l2_weight_decay=1e-4 \
        --train_batch_size=$BATCH_SIZE \
        --eval_batch_size=1 \
        --dist_eval \
        --training_file_pattern="/home/ubuntu/data/nv_coco/train*.tfrecord" \
        --validation_file_pattern="/home/ubuntu/data/nv_coco/val*.tfrecord" \
        --val_json_file="/home/ubuntu/data/annotations/instances_val2017.json" \
        --amp \
        --use_batched_nms \
        --xla \
        --tf2 \
        --async_eval \
        --use_ext \
        --use_custom_box_proposals_op | tee $BASEDIR/../baseline_1x/baseline_1x.log