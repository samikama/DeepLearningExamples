#!/bin/bash

source activate tensorflow2_latest_p37

horovodrun -np 8 --autotune python train_tf2.py
  
mpirun -np 8 \
--H localhost:8 \
--allow-run-as-root \
--mca plm_rsh_no_tree_spawn 1 -bind-to none -map-by slot -mca pml ob1 -mca btl ^openib \
-mca btl_tcp_if_exclude lo,docker0 \
-mca btl_vader_single_copy_mechanism none \
-x LD_LIBRARY_PATH \
-x PATH \
-x NCCL_SOCKET_IFNAME=^docker0,lo \
-x NCCL_MIN_NRINGS=8 \
-x TF_CUDNN_USE_AUTOTUNE=0 \
-x TF_ENABLE_NHWC=1 \
-x HOROVOD_CYCLE_TIME=0.5 \
-x HOROVOD_FUSION_THRESHOLD=67108864 \
python train_tf2.py