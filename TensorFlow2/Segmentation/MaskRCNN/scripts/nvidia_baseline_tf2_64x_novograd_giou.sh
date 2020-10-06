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

conda_path=/shared/conda2
source $conda_path/etc/profile.d/conda.sh
conda activate base

BASEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
rm -rf $BASEDIR/../results_tf2_64x_novo_$1
mkdir -p $BASEDIR/../results_tf2_64x_novo_$1
/opt/amazon/openmpi/bin/mpirun --allow-run-as-root --tag-output --mca plm_rsh_no_tree_spawn 1 \
    --mca btl_tcp_if_exclude lo,docker0 \
    --hostfile /shared/hostfiles/hosts_64x \
    -N 8 \
    -x NCCL_DEBUG=VERSION \
    -x LD_LIBRARY_PATH \
    -x PATH \
    --oversubscribe \
    /shared/conda2/bin/python ${BASEDIR}/../mask_rcnn_main.py \
        --mode="train_and_eval" \
	--loop_mode="tape" \
	--box_loss_type="giou" \
        --checkpoint="/shared/DeepLearningExamples.ben/TensorFlow2/Segmentation/MaskRCNN/weights/resnet/resnet-nhwc-2018-02-07/model.ckpt-112603" \
        --eval_samples=5000 \
        --log_interval=10 \
        --init_learning_rate=0.08 \
        --optimizer_type="Novograd" \
        --lr_schedule="cosine" \
        --model_dir="$BASEDIR/../results_tf2_64x_novo_$1" \
        --num_steps_per_eval=231 \
        --warmup_learning_rate=0.000133 \
	--beta1=0.9 \
	--beta2=0.25 \
	--warmup_steps=1000 \
        --total_steps=5082 \
        --l2_weight_decay=1.25e-3 \
        --train_batch_size=1 \
        --eval_batch_size=1 \
        --dist_eval \
	--first_eval=15 \
        --training_file_pattern="/shared/data2/train*.tfrecord" \
        --validation_file_pattern="/shared/data2/val*.tfrecord" \
        --val_json_file="/shared/data2/annotations/instances_val2017.json" \
        --amp \
        --use_batched_nms \
        --xla \
        --tf2 \
        --use_custom_box_proposals_op | tee $BASEDIR/../results_tf2_64x_novo_$1/train_eval.log
