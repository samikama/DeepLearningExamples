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

# Batch size per gpu 4 on arbitrary number of nodes
# as specified in hosts file

BATCH_SIZE=1
HOST_COUNT=`wc -l < /shared/ben/hosts`
#GPU_COUNT=`nvidia-smi --query-gpu=name --format=csv,noheader | wc -l`
GPU_COUNT=8
IMAGES=118287
GLOBAL_BATCH_SIZE=$((BATCH_SIZE * HOST_COUNT * GPU_COUNT))
STEP_PER_EPOCH=$(( IMAGES / GLOBAL_BATCH_SIZE ))
FIRST_DECAY=5625
SECOND_DECAY=7500
TOTAL_STEPS=8500
BASE_LR=0.24
WARMUP_STEPS=1800

source /shared/conda/bin/activate mask_rcnn

BASEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
rm -rf $BASEDIR/../results_tape_1x
mkdir -p $BASEDIR/../results_tape_1x
/opt/amazon/openmpi/bin/mpirun --allow-run-as-root --tag-output --mca plm_rsh_no_tree_spawn 1 \
    -x RDMAV_FORK_SAFE=1 \
    --mca btl_tcp_if_exclude lo,docker0 \
    --hostfile /shared/ben/hosts \
    -x NCCL_DEBUG=VERSION \
    -x LD_LIBRARY_PATH \
    -x PATH \
    --oversubscribe \
    /shared/conda/envs/mask_rcnn/bin/python ${BASEDIR}/../mask_rcnn_main.py \
        --mode="train_and_eval" \
        --checkpoint="/shared/ben/DeepLearningExamples/TensorFlow2/Segmentation/MaskRCNN/resnet/resnet-nhwc-2018-02-07/model.ckpt-112603" \
        --eval_samples=5000 \
        --loop_mode="tape" \
        --log_interval=100 \
        --init_learning_rate=0.24 \
        --learning_rate_steps="5625,7500" \
        --optimizer_type="SGD" \
        --lr_schedule="piecewise" \
        --model_dir="$BASEDIR/../results_tape_1x" \
        --num_steps_per_eval=$STEP_PER_EPOCH \
        --warmup_learning_rate=0.000133 \
        --warmup_steps=1800 \
	--first_eval=7 \
        --global_gradient_clip_ratio=0.0 \
        --total_steps=8500 \
        --l2_weight_decay=1e-4 \
        --train_batch_size=$BATCH_SIZE \
        --eval_batch_size=1 \
        --dist_eval \
        --training_file_pattern="/shared/data/coco/nv_tfrecords/train*.tfrecord" \
        --validation_file_pattern="/shared/data/coco/nv_tfrecords/val*.tfrecord" \
        --val_json_file="/shared/data/coco/nv_tfrecords/annotations/instances_val2017.json" \
        --amp \
        --xla \
        --use_batched_nms \
        --async_eval \
        --use_ext \
        --use_custom_box_proposals_op | tee $BASEDIR/../results_tape_1x/results_tape_1x.log
