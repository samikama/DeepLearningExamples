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

DIRECT_LAUNCH=${DIRECT_LAUNCH:-"0"}
WITH_XLA=${WITH_XLA:-1}
BATCH_SIZE=1
HOST_COUNT=1
NUM_GPUS=8
PRECALC_DATASET=${PRECALC_DATASET:-1}

BASEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
rm -rf $BASEDIR/../results_tf2_64x_novo_$1
mkdir -p $BASEDIR/../results_tf2_64x_novo_$1
 

/shared/rejin/conda/bin/herringrun -n 64 -c /shared/rejin/conda \
    RUN_HERRING=1 \
    /shared/rejin/conda/bin/python  ${BASEDIR}/bind_launch.py  --direct_launch=${DIRECT_LAUNCH} --nproc_per_node=${NUM_GPUS} --nsockets_per_node=2 --ncores_per_socket=24 ${BASEDIR}/../mask_rcnn_main.py \
        --mode="train_and_eval" \
	--loop_mode="tape" \
	--box_loss_type="giou" \
        --checkpoint="/shared/rejin/DeepLearningExamples/TensorFlow2/Segmentation/MaskRCNN/resnet/resnet-nhwc-2018-02-07/model.ckpt-112603" \
        --eval_samples=5000 \
        --log_interval=10 \
        --init_learning_rate=0.07 \
        --optimizer_type="Novograd" \
        --lr_schedule="cosine" \
        --model_dir="$BASEDIR/../results_tf2_64x_novo_$1" \
        --num_steps_per_eval=231 \
        --warmup_learning_rate=0.000133 \
	--beta1=0.9 \
	--beta2=0.25 \
	--warmup_steps=1000 \
        --total_steps=4158 \
        --l2_weight_decay=1.25e-3 \
	--label_smoothing=0.1 \
        --train_batch_size=1 \
        --eval_batch_size=1 \
        --dist_eval \
	--first_eval=22 \
	--use_carl_loss \
        --training_file_pattern="/scratch/precalc_masks_latest/train*.tfrecord" \
        --validation_file_pattern="/shared/data2/val*.tfrecord" \
        --val_json_file="/shared/data2/annotations/instances_val2017.json" \
        --amp \
        --use_batched_nms \
        --xla \
        --tf2 \
	--preprocessed_data=${PRECALC_DATASET} \
        --use_custom_box_proposals_op | tee $BASEDIR/../results_tf2_64x_novo_$1/train_eval.log
