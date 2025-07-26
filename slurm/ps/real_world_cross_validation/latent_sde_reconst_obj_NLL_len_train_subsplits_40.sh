#!/bin/bash

cd /home/seifner/repos/FIM/ || exit
source /home/seifner/repos/FIM/.venv/sw/bin/activate || exit
gpustat

base="python3 scripts/sde/train_latent_sde_on_real_world_cross_validation.py --base-config=configs/train/sde/20250726_latent_sde_real_world_setup.yaml \
    --seed=0 \
    --exp-name=latent_sde_latent_dim_4_context_dim_100_decoder_NLL_len_train_subsplits_40 \
    --epochs-to-evaluate=499,999,1999,4999 \
    --save-every=500 \
    --epochs=5000 \
    --len-train-subsplits=40 \
    --mse-objective=False \
"

eval "${base} --dataset-label=tsla --split=0" &
sleep 15
eval "${base} --dataset-label=fb --split=0" &
sleep 15
eval "${base} --dataset-label=wind --split=0" &
sleep 15
eval "${base} --dataset-label=oil --split=0" &
sleep 15
eval "${base} --dataset-label=tsla --split=1" &
sleep 15
eval "${base} --dataset-label=fb --split=1" &
sleep 15
eval "${base} --dataset-label=wind --split=1" &
sleep 15
eval "${base} --dataset-label=oil --split=1" &
sleep 15
eval "${base} --dataset-label=tsla --split=2" &
sleep 15
eval "${base} --dataset-label=fb --split=2" &
sleep 15
eval "${base} --dataset-label=wind --split=2" &
sleep 15
eval "${base} --dataset-label=oil --split=2" &
sleep 15
eval "${base} --dataset-label=tsla --split=3" &
sleep 15
eval "${base} --dataset-label=fb --split=3" &
sleep 15
eval "${base} --dataset-label=wind --split=3" &
sleep 15
eval "${base} --dataset-label=oil --split=3" &
sleep 15
eval "${base} --dataset-label=tsla --split=4" &
sleep 15
eval "${base} --dataset-label=fb --split=4" &
sleep 15
eval "${base} --dataset-label=wind --split=4" &
sleep 15
eval "${base} --dataset-label=oil --split=4"
sleep 15
