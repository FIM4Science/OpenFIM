# FIM

<div align="center">
  <a href="https://github.com/cvejoski/FIM/actions/workflows/ci.yml">
    <img src="https://github.com/cvejoski/FIM/actions/workflows/ci.yml/badge.svg" alt="CI">
  </a>
  <a href="https://github.com/cvejoski/FIM/blob/main/LICENSE.txt">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
  </a>
</div>

The Foundation Inference Model (FIM) library offers a streamlined implementation of the FIM methodology, including models, training procedures, and example inference scripts. Built on PyTorch, the library simplifies the process of training and evaluating FIMs. Instead of writing models and training routines from scratch, users can define configurations in a simple .yaml file, enabling quick experimentation to tackle complex problems.
The library originates from the FIM series of papers ([References](#references)). Pretrained models that replicate the results from these publications are available on [Hugging Face](https://huggingface.co/FIM4Science). A [tutorial](notebooks/tutorials/fim-mjp.ipynb) is also provided to guide users through these features.


## Table of Contents
- FIM
  - [Installation](#installation)
  - [Usage](#usage)
  - [Configuration File Explanation](#configuration-file-explanation)
  - [Models](#models)
    - [Markov Jump Process](#markov-jump-process)
  - [Contributing](#contributing)
  - [Lamarr’s DL4SD lab](#lamarrs-dl4sd-lab)
  - [License](#license)
  - [References](#references)

## Installation

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

Then take a look into the `scripts` and `notebooks` folders.

## Usage

To start training, follow these steps:

1. Make sure you have activated the virtual environment (see Installation).

2. Create a configuration file in YAML format, e.g.,

config.yaml

, with the necessary parameters for training.

3. Run the training script in single-node mode, providing the path to the configuration file:

   ```bash
   python scripts/train_model.py --config configs/train/example.yaml
   ```

   This will start the training process using the specified configuration and save the trained model to the specified location.

4. To start training in distributed mode using `torchrun`, use the following command:

   ```bash
   torchrun --nproc_per_node=<number_of_gpus> scripts/train_model.py --config configs/train/example.yaml
   ```

   Replace `<number_of_gpus>` with the number of GPUs you want to use for distributed training.

5. Monitor the training progress and adjust the parameters in the configuration file as needed.

## Configuration File Explanation

The configuration file is a YAML file that contains all the necessary parameters for training. Below is an explanation of the key sections in the configuration file:

### Experiment Section

```yaml
experiment:
  name: FIM_MJP_Homogeneous_no_annealing_rnn_256_path_attention_one_head_model_dim_var_path_same
  name_add_date: true # if true, the current date & time will be added to the experiment name
  seed: [0]
  device_map: auto # auto, cuda, cpu
```
- `name`: The name of the experiment.
- `name_add_date`: If true, the current date & time will be added to the experiment name.
- `seed`: The seed for random number generation to ensure reproducibility.
- `device_map`: The device to use for training. Options are `auto`, `cuda`, and `cpu`.

### Distributed Section

```yaml
distributed:
  enabled: true
  sharding_strategy: NO_SHARD # SHARD_GRAD_OP, NO_SHARD, HYBRID_SHARD
  wrap_policy: SIZE_BAZED # NO_POLICY, MODEL_SPECIFIC, SIZE_BAZED
  min_num_params: 1e5
  checkpoint_type: full_state # full_state, local_state
  activation_chekpoint: false
```

- `enabled`: Whether to enable distributed training.
- `sharding_strategy`: The sharding strategy to use. Options are `SHARD_GRAD_OP`, `NO_SHARD`, and `HYBRID_SHARD`.
- `wrap_policy`: The policy for wrapping layers. Options are `NO_POLICY`, `MODEL_SPECIFIC`, and `SIZE_BASED`.
- `min_num_params`: The minimum number of parameters for size-based wrapping.
- `checkpoint_type`: The type of checkpoint to use. Options are `full_state` and `local_state`.
- `activation_checkpoint`: Whether to enable activation checkpointing.

### Dataset Section

```yaml
dataset:
  name: FIMDataLoader
  path_collections:
    train: !!python/tuple
      - /path/to/train/data1
      - /path/to/train/data2
    validation: !!python/tuple
      - /path/to/validation/data1
      - /path/to/validation/data2
  loader_kwargs:
    batch_size: 128
    num_workers: 16
    test_batch_size: 128
    pin_memory: true
    max_path_count: 300
    max_number_of_minibatch_sizes: 10
    variable_num_of_paths: true
  dataset_kwargs:
    files_to_load:
      observation_grid: "fine_grid_grid.pt"
      observation_values: "fine_grid_noisy_sample_paths.pt"
      mask_seq_lengths: "fine_grid_mask_seq_lengths.pt"
      time_normalization_factors: "fine_grid_time_normalization_factors.pt"
      intensity_matrices: "fine_grid_intensity_matrices.pt"
      adjacency_matrices: "fine_grid_adjacency_matrices.pt"
      initial_distributions: "fine_grid_initial_distributions.pt"
    data_limit: null
```

- `name`: The name of the data loader.
- `path_collections`: Paths to the training and validation data.
- `loader_kwargs`: Additional arguments for the data loader, such as batch size, number of workers, etc.
- `dataset_kwargs`: Additional arguments for the dataset, such as files to load and data limit.

### Model Section

```yaml
model:
  model_type: fimmjp
  n_states: 6
  use_adjacency_matrix: false
  ts_encoder:
    name: fim.models.blocks.base.RNNEncoder
    rnn:
      name: torch.nn.LSTM
      hidden_size: 256
      batch_first: true
      bidirectional: true
  pos_encodings:
    name: fim.models.blocks.positional_encodings.DeltaTimeEncoding
  path_attention:
    name: fim.models.blocks.MultiHeadLearnableQueryAttention
    n_queries: 16
    n_heads: 1
    embed_dim: 512
    kv_dim: 128
  intensity_matrix_decoder:
    name: fim.models.blocks.base.MLP
    in_features: 2049
    hidden_layers: !!python/tuple [128, 128]
    hidden_act:
      name: torch.nn.SELU
    dropout: 0
    initialization_scheme: lecun_normal
  initial_distribution_decoder:
    name: fim.models.blocks.base.MLP
    in_features: 2049
    hidden_layers: !!python/tuple [128, 128]
    hidden_act:
      name: torch.nn.SELU
    dropout: 0
    initialization_scheme: lecun_normal
```

- `model_type`: The type of model to use.
- `n_states`: The number of states in the Markov jump process.
- `use_adjacency_matrix`: Whether to use an adjacency matrix.
- `ts_encoder`: Configuration for the time series encoder.
- `pos_encodings`: Configuration for the positional encodings.
- `path_attention`: Configuration for the path attention mechanism.
- `intensity_matrix_decoder`: Configuration for the intensity matrix decoder.
- `initial_distribution_decoder`: Configuration for the initial distribution decoder.

### Trainer Section

```yaml
trainer:
  name: Trainer
  debug_iterations: null
  precision: bf16 # null, fp16, bf16, bf16_mixed, fp16_mixed, fp32_policy
  epochs: 3000
  detect_anomaly: false
  save_every: 10
  gradient_accumulation_steps: 1
  best_metric: loss
  logging_format: "RANK_%(rank)s - %(asctime)s - %(name)s - %(levelname)s - %(message)s"
  experiment_dir: ./results/
  schedulers: !!python/tuple
    - name: fim.utils.param_scheduler.ConstantScheduler
      beta: 1.0
      label: gauss_nll
    - name: fim.utils.param_scheduler.ConstantScheduler
      label: init_cross_entropy
      beta: 1.0
    - name: fim.utils.param_scheduler.ConstantScheduler
      label: missing_link
      beta: 1.0
```

- `name`: The name of the trainer.
- `debug_iterations`: Number of debug iterations.
- `precision`: The precision to use for training. Options are `null`, `fp16`, `bf16`, `bf16_mixed`, `fp16_mixed`, and `fp32_policy`.
- `epochs`: The number of epochs to train for.
- `detect_anomaly`: Whether to detect anomalies during training.
- `save_every`: Save the model every specified number of epochs.
- `gradient_accumulation_steps`: Number of gradient accumulation steps.
- `best_metric`: The metric to use for determining the best model.
- `logging_format`: The format for logging messages.
- `experiment_dir`: The directory to save experiment results.
- `schedulers`: Configuration for the schedulers.

### Optimizers Section

```yaml
optimizers: !!python/tuple
  - optimizer_d:
      name: torch.optim.AdamW
      lr: 0.00001
      weight_decay: 0.0001
```

- `optimizers`: Configuration for the optimizers.

## Models

Currently, the following models are implemented:

  - [Markov Jump Process](#markov-jump-process)

### Markov Jump Process

This model is based on the paper:

- David Berghaus, Kostadin Cvejoski, Patrick Seifner, Cesar Ojeda, Ramses J. Sanchez, "Foundation Inference Models for Markov Jump Processes," 2024. [OpenReview](https://openreview.net/forum?id=f4v7cmm5sC).

Markov jump processes are continuous-time stochastic processes which describe dynamical systems evolving in discrete state spaces. These processes find wide application in the natural sciences and machine learning, but their inference is known to be far from trivial. In this work we introduce a methodology for zero-shot inference of Markov jump processes (MJPs), on bounded state spaces, from noisy and sparse observations, which consists of two components. First, a broad probability distribution over families of MJPs, as well as over possible observation times and noise mechanisms, with which we simulate a synthetic dataset of hidden MJPs and their noisy observation process. Second, a neural network model that processes subsets of the simulated observations, and that is trained to output the initial condition and rate matrix of the target MJP in a supervised way. We empirically demonstrate that one and the same (pretrained) model can infer, in a zero-shot fashion, hidden MJPs evolving in state spaces of different dimensionalities. Specifically, we infer MJPs which describe (i) discrete flashing ratchet systems, which are a type of Brownian motors, and the conformational dynamics in (ii) molecular simulations, (iii) experimental ion channel data and (iv) simple protein folding models. What is more, we show that our model performs on par with state-of-the-art models which are finetuned to the target datasets.

> For a detailed tutorial on how to use the FIM-MJP model, please refer to the [Jupyter Notebook](notebooks/tutorials/fim-mjp.ipynb). The model is also available on [Hugging Face](https://huggingface.co/FIM4Science/fim-mjp).




## Contributing

Contributions are welcome! Here's how you can contribute to this project:

1. Create a new branch:

   ```bash
   git checkout -b feature/your-feature
   ```

2. Make your changes and commit them:

   ```bash
   git commit -m "Add your commit message here"
   ```

3. Push your changes to your forked repository:

   ```bash
   git push origin feature/your-feature
   ```

4. Open a pull request.

## Lamarr’s DL4SD lab

Lamarr’s Deep Learning for Scientific Discovery (DL4SD) lab is an interdisciplinary team of researchers working at the intersection of machine learning, statistical physics, and complexity science, to develop neural systems that automatically construct scientific hypotheses — articulated as mathematical models — to explain complex natural and social phenomena.

To achieve this overarching goal, we design pre-trained neural recognition models that encode classical mathematical models commonly used in the natural and social sciences. And focus on mathematical models that are simple enough to remain approximately valid across a wide range of observation scales, from microscopic to coarse-grained.

Fundamentally, these pre-trained neural recognition models enable the zero-shot inference of (the parameters defining) the mathematical equations they encode directly from data. We refer to these models as Foundation Inference Models (FIMs).

## License

This project is licensed under the [MIT License](LICENSE).

## References

- David Berghaus, Kostadin Cvejoski, Patrick Seifner, Cesar Ojeda, Ramses J. Sanchez, "Foundation Inference Models for Markov Jump Processes" NeurIPS 2025. [OpenReview](https://openreview.net/forum?id=f4v7cmm5sC)
- Patrick Seifner, Kostadin Cvejoski, Antonia Körner, Ramsés J. Sánchez, "Foundational Inference Models for Dynamical Systems
", [arxiv](https://arxiv.org/abs/2402.07594v2)
