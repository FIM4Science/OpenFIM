#!/bin/bash

#SBATCH --job-name=Llama2-7B-CSQA

#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus-per-task=4

export LOGLEVEL=INFO
# debugging flags (optional)
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=WARN
export PYTHONFAULTHANDLER=1
# export LD_LIBRARY_PATH=/opt/amazon/efa/lib:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH
export CUDA_LAUNCH_BLOCKING=0

# on your cluster you might need these:
# set the network interface
export NCCL_SOCKET_IFNAME="ens"
export FI_EFA_USE_DEVICE_RDMA=1
workon reasoning
srun torchrun --nproc-per-node=4 scripts/train_model.py -cc configs/train/llama2_commonsenseqa.yaml

