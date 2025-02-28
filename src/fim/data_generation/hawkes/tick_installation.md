0.
   ```bash
   srun --mem=64GB --export ALL -c 16 --container-name=tick_ubuntu --job-name="3_state_1_percent_equilibrium" --container-image=nvcr.io/ml2r/interactive_ubuntu:23.04 --mail-user=david.berghaus@iais.fraunhofer.de --mail-type=ALL --pty /bin/bash
   ```

1.
   ```bash
   conda create -n "py36tick" python=3.6
   ```

2.
   ```bash
  conda activate py36tick
   ```

3.
   ```bash
   pip install /cephfs_projects/foundation_models/tick-0.7.0.1-cp36-cp36m-linux_x86_64.whl
   ```
