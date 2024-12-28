#!/bin/bash

USER_NODE="burela"
ENTRY_SERVER="cluster-g.math.tu-berlin.de"

GPU_NAME="a100_pcie_80g"
NUM_GPUS="0"
MEM="10G"
NUM_CPUS="1"

echo "Login to entry server for allocation ( cluster PW )"
ssh ${USER_NODE}@${ENTRY_SERVER} -t "
salloc -p gbr --gres=gpu:${GPU_NAME}:${NUM_GPUS} --mem=${MEM} &
srun --pty -p gbr --gres=gpu:${GPU_NAME}:${NUM_GPUS} --mem=${MEM} bash --login 
"
