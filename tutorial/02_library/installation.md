# Installation

In order to set up the necessary environment using [uv](https://docs.astral.sh/uv/):

```bash
uv sync --python 3.12

source .venv/bin/activate
```

For contributions to the library, additionally install pre-commit hooks:

```bash
pre-commit install
pre-commit autoupdate
```

Check out the configuration under `.pre-commit-config.yaml`. The `-n, --no-verify` flag of `git commit` can be used to deactivate pre-commit hooks temporarily.

## Usage


```{note}
If you want to train your own models and confirm that everything is installed properly use the following steps. Otherwise
if you want to use our trained models you can safely skip this part.
```

To start training, follow these steps:

1. Make sure you have activated the virtual environment (see Installation).

2. Create a [configuration file](configuration.md), e.g. `your-config.yaml`, with the necessary parameters for training.

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

5. Monitor the training progress on tensorboard and adjust the parameters in the configuration file as needed.
