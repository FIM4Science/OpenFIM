#!/bin/bash

date
echo "Job Started on"
hostname
echo "Slurm job id: ${SLURM_JOB_ID}"

cd /home/seifner/repos/FIM/ || exit
source /home/seifner/repos/FIM/.venv/sw/bin/activate || exit
gpustat
sleep 5
eval "$1"
