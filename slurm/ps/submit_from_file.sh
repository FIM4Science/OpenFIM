#!/bin/bash

# exectute from slurm_scripts directory on login node
# ps_run_in_fim_venv.sh must be exectutable and in a container

# pass txt file name with commands
file_with_cmds=$1
stripped_filename=${file_with_cmds%.txt}
filename_start=$(echo $file_with_cmds | cut -c1-5)

exec 3<"$file_with_cmds" # open in file descriptor 3 (0-2 are reserved for stdin stdout stderr)

itr=1
while IFS= read -r line <&3; do
	echo "Started in ${itr}: $line"
	srun_params="--mem=32GB -c 16 --partition=GPU1 --gres=gpu:1 --export ALL \
    --container-name='sde_torch_23_12' --container-image=nvcr.io/ml2r/interactive_pytorch:23.12-py3 --container-workdir=$HOME \
    --mail-user=seifner@iai.uni-bonn.de --mail-type=ALL \
    -e ~/slurm_logs/${stripped_filename}-${itr}-%j.err \
    -o ~/slurm_logs/${stripped_filename}-${itr}-%j.out"

	eval "srun ${srun_params} --job-name=${filename_start}_${itr} $HOME/slurm_scripts/ps_run_in_fim_venv.sh '${line}'" # &
	((itr += 1))
done
exec 3<&- # close file descriptor 3
