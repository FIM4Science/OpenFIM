# Installation
In order to set up the necessary environment:

1. Create a virtual environment using your conda or python virtualenv:

   ```bash
   conda create -n fim_env python=3.12
   conda activate fim_env
   ```

2. Install the project in the virtual environment:

   ```bash
   pip install -e .
   ```

Optional and needed only once after `git clone`:

3. Install several [pre-commit] git hooks with:

   ```bash
   pre-commit install
   # You might also want to run `pre-commit autoupdate`
   ```

   and check out the configuration under `.pre-commit-config.yaml`. The `-n, --no-verify` flag of `git commit` can be used to deactivate pre-commit hooks temporarily.

Then take a look into the `scripts` folder.

## Usage


```{note}
If you want to train your own models and confirm that everything is installed properly use the following steps. Otherwise
if you want to use our trained models you can safely skip this part.
```

To start training, follow these steps:

1. Make sure you have activated the virtual environment (see Installation).

2. Create a configuration file in YAML format, e.g.,

your-config.yaml

, with the necessary parameters for training.

3. Run the training script in single-node mode, providing the path to the configuration file:

   ```bash
   python scripts/train_model.py --config <path/to/your-config.yaml>
   ```

   This will start the training process using the specified configuration and save the trained model to the specified location.

4. To start training in distributed mode using `torchrun`, use the following command:

   ```bash
   torchrun --nproc_per_node=<number_of_gpus> scripts/train_model.py --config <path/to/your-config.yaml>
   ```

   Replace `<number_of_gpus>` with the number of GPUs you want to use for distributed training.

5. Monitor the training progress and adjust the parameters in the configuration file as needed.
