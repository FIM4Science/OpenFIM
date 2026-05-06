#!/bin/bash

# train data defined in yaml
# generate each setup in a separate srun
# number of parallel executions limited by user

max_parallel_sruns=6
wait_time=10
max_total_runs=14
generation_label="600k_ode_corrupted_dim_1_2_3"
project_path="/home/seifner/repos/FIM"
data_path="/home/seifner/repos/FIM/data"
yaml_path="configs/data_generation/ode/ode_600k_deg_3_dim_1_2_3_monomial_survival_uniform_corrupted.yaml"

# declare -a nodes=("ml2ran02" "ml2ran03" "ml2ran05" "ml2ran06" "ml2ran08" "ml2ran11")
itr=0

for itr in $(seq 0 $max_total_runs); do
	sleep 10 # wait a bit before previous jobs has started to capture it in squeue --me

	num_jobs=$(($(squeue --me | wc -l) - 1))
	while [ $num_jobs -ge $max_parallel_sruns ]; do
		echo -ne "At maximum of parallel jobs: " $num_jobs "-> waiting " $wait_time \\r
		sleep $wait_time
		num_jobs=$(($(squeue --me | wc -l) - 1))
	done

	# node_id=$((itr % 5))
	# -w ${nodes[$node_id]}

	echo "Started setup index: ${itr}"
	srun_params="--mem=128GB --export ALL -c 32 --partition=CPU \
    --container-name='seifner_pytorch_ode' --container-image=nvcr.io/ml2r/interactive_pytorch:23.12-py3 --container-workdir=$HOME \
    --mail-user=seifner@iai.uni-bonn.de --mail-type=ALL \
    --exclude=ml2ran07 \
    -e ~/slurm_logs/${generation_label}-${itr}-%j.err \
    -o ~/slurm_logs/${generation_label}-${itr}-%j.out"

	cmd="python3 scripts/sde/data_generation/train/generate_one_setup_in_yaml.py \
        --generation_label=${generation_label} \
        --project_path=${project_path} \
        --data_path=${data_path} \
        --yaml_path=${yaml_path} \
        --index=${itr}"

	eval "srun ${srun_params} --job-name=${generation_label}_${itr} $HOME/slurm_scripts/ps/run_in_fim_venv.sh '${cmd}'" &
	((itr += 1))
done
