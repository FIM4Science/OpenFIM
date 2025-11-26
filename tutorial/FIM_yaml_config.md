# Building your own configuration file

The configuration file is a YAML file that contains all the necessary parameters for training. Below is an explanation of the key sections in the configuration file:

## Experiment Section

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

## Distributed Section

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

## Dataset Section

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

## Model Section

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
