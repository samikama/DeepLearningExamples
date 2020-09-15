#!/bin/bash

source activate tensorflow2_latest_p37

mpirun --allow-run-as-root \
    -np `nvidia-smi --query-gpu=name --format=csv,noheader | wc -l` \
    python train.py