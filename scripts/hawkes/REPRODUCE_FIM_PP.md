# Reproducing the FIM-PP results (ICLR 2026)

This page explains how to reproduce the real-data results of
["In-Context Learning of Temporal Point Processes with Foundation Inference Models"](https://openreview.net/forum?id=h9HwUAODFP) (ICLR 2026):
Table 1 (multi-event forecasting, N = 20), the appendix tables for N = 5 and N = 10, and Table 2 (next-event prediction).
The scripts are the ones that produced the paper numbers, and the checkpoints are the exact weights that were evaluated.

| File | Purpose |
|---|---|
| `scripts/hawkes/fim_benchmark.py` | Runner: evaluates a list of checkpoints on a list of datasets, optionally fine-tunes first, and writes `summary.csv`, per-metric matrices and LaTeX rows |
| `scripts/hawkes/fim_long_horizon_prediction.py` | Multi-event forecasting (Table 1): autoregressive rollout, OTD, RMSE<sub>e</sub>, RMSE<sub>Δt</sub>, sMAPE<sub>Δt</sub> |
| `scripts/hawkes/otd_metrics.py` | Optimal-transport distance (OTD) between event sequences, following the CDiff / HYPRO implementation |
| `scripts/hawkes/fim_next_event_prediction.py` | Next-event prediction (Table 2): RMSE<sub>Δt</sub>, mark accuracy, sMAPE<sub>Δt</sub> |
| `scripts/hawkes/fim_finetune.py` | Fine-tuning on a dataset's train split (called by the runner when `fine_tune: true`) |
| `scripts/hawkes/check_cdiff_data.py` | Checks that the CDiff splits are in place and identical to the ones used in the paper |
| `scripts/hawkes/configs/fim_pp_iclr2026/*.yaml` | One runner config per table row |

## 1. Setup

```bash
uv sync --python 3.12
source .venv/bin/activate
```

All commands below are run from the repository root. Each evaluation uses one GPU.

## 2. Data

**Table 1 (multi-event) uses the CDiff splits** of Zeng et al. (2024). Download them from the
[CDiff data folder](https://drive.google.com/drive/folders/1gT3fL5vJpLYPNtn9eGAbC7qrhmFgiyV0) (linked from the
[CDiff repository](https://github.com/networkslab/cdiff)) and arrange them as

```
data/external/CDiff_dataset/
├── taxi/{train,dev,test}.pkl
├── taobao/{train,dev,test}.pkl
├── stackoverflow/{train,dev,test}.pkl   # "so/" in the CDiff folder
├── amazon/{train,dev,test}.pkl          # also in the CDiff GitHub repo under data/amazon
└── retweet/{train,dev,test}.pkl
```

Then check that the files match the ones used in the paper:

```bash
python scripts/hawkes/check_cdiff_data.py
```

| Dataset | train / dev / test sequences | Marks |
|---|---|---|
| Taxi | 1400 / 200 / 400 | 10 |
| Taobao | 1300 / 200 / 500 | 17 |
| StackOverflow | 1401 / 401 / 401 | 22 |
| Amazon | 5200 / 300 / 500 | 16 |
| Retweet | 6500 / 731 / 500 | 3 |

The loader shifts every sequence so its first event is at t = 0. The CDiff and EasyTPP releases of these datasets are not interchangeable (Amazon, for example, has different split sizes), so use the CDiff files.

**Table 2 (next-event) uses the EasyTPP datasets on the Hugging Face Hub** (`easytpp/taxi`, `easytpp/taobao`).
They are downloaded automatically. The runs used revisions `31af59ee1ae184b5b7abbf0094099b1172d66e2a` (taxi) and
`a96ec6dc7ef311ca45993af9c8a80f78ed08ae47` (taobao), which have been the latest revisions since 2024.
The dataset analysis figures in the paper appendix ("EasyTPP Dataset Analysis") were made from these EasyTPP files.

## 3. Checkpoints

All checkpoints are on the Hugging Face Hub, and every script accepts a Hub id wherever it takes a checkpoint directory.

| Paper result | Checkpoint |
|---|---|
| Table 1 and appendix N = 5/10, FIM-PP (zs) | `FIM4Science/fim-pp` (epoch 36 of pretraining) |
| Table 1 and appendix N = 5/10, FIM-PP (f) | `FIM4Science/fim-pp-paper-checkpoints/table1-finetuned/<dataset>` |
| Table 2, FIM-PP (zs) | `FIM4Science/fim-pp-paper-checkpoints/table2-zero-shot` (epoch 38 of the same pretraining run) |
| Table 2, FIM-PP (f) | `FIM4Science/fim-pp-paper-checkpoints/table2-finetuned/<dataset>` |

Epochs 36 and 38 have almost the same validation loss (−0.4750 vs −0.4757).
Table 1 was evaluated while pretraining was still running, and Table 2 after it finished. That is why the two tables use different epochs.
The Table 1 fine-tunes were trained on the CDiff train splits, and the Table 2 fine-tunes on the EasyTPP train splits.

## 4. Running the evaluations

```bash
python scripts/hawkes/fim_benchmark.py --config scripts/hawkes/configs/fim_pp_iclr2026/table1_zero_shot.yaml
python scripts/hawkes/fim_benchmark.py --config scripts/hawkes/configs/fim_pp_iclr2026/table1_finetuned.yaml
python scripts/hawkes/fim_benchmark.py --config scripts/hawkes/configs/fim_pp_iclr2026/table2_zero_shot.yaml
python scripts/hawkes/fim_benchmark.py --config scripts/hawkes/configs/fim_pp_iclr2026/table2_finetuned.yaml
```

Each run writes to `results/fim_pp_iclr2026/<config>/<timestamp>/`:
- one folder per dataset, with `metrics.json`, `config_used.json` and `run.log`
- `summary.csv`
- one CSV per metric
- the LaTeX rows

For the appendix tables, set `forecast_horizon_size` to 5 or 10.
Set `parallel` to evaluate several datasets at once.

Single-dataset runs can also call the evaluation scripts directly, for example:

```bash
python scripts/hawkes/fim_long_horizon_prediction.py --checkpoint FIM4Science/fim-pp --dataset taxi \
  --run-dir results/taxi_zs --forecast-horizon-size 20 --context-size 2000 --max-num-events 100 \
  --num-ensemble-trajectories 5 --num-trials 3 --base-seed 0 --sampling-method thinning
```

### Expected results

Multi-event forecasting, N = 20: mean ± std over 3 trials, as produced by the runs behind Table 1.
Taobao is not in Table 1 of the paper.
The run time is the wall-clock time per dataset on one GPU for all 3 trials.

| Dataset | Model | OTD | RMSE<sub>e</sub> | RMSE<sub>Δt</sub> | sMAPE<sub>Δt</sub> | Run time |
|---|---|---|---|---|---|---|
| Taxi | zs | 23.145 ± 0.073 | 1.421 ± 0.014 | 0.277 ± 0.000 | 76.765 ± 0.386 | 4.4 h |
| Taxi | f | 17.914 ± 0.117 | 0.705 ± 0.006 | 0.314 ± 0.004 | 76.828 ± 0.549 | 4.9 h |
| Taobao | zs | 64.281 ± 0.077 | 3.949 ± 0.010 | 1.988 ± 0.006 | 169.687 ± 0.089 | 5.4 h |
| Taobao | f | 60.106 ± 0.464 | 2.428 ± 0.005 | 16.068 ± 0.109 | 152.528 ± 0.377 | 5.9 h |
| StackOverflow | zs | 49.259 ± 0.056 | 2.393 ± 0.015 | 1.068 ± 0.002 | 96.364 ± 0.048 | 4.8 h |
| StackOverflow | f | 39.792 ± 0.042 | 1.336 ± 0.030 | 1.018 ± 0.003 | 88.248 ± 0.189 | 4.8 h |
| Amazon | zs | 46.219 ± 0.108 | 2.073 ± 0.012 | 0.464 ± 0.001 | 128.635 ± 0.398 | 5.8 h |
| Amazon | f | 37.208 ± 0.098 | 2.030 ± 0.019 | 0.366 ± 0.001 | 81.188 ± 0.142 | 5.6 h |
| Retweet | zs | 60.238 ± 0.161 | 4.172 ± 0.064 | 24.057 ± 0.050 | 99.069 ± 0.390 | 5.9 h |
| Retweet | f | 59.437 ± 0.082 | 2.703 ± 0.012 | 21.985 ± 0.014 | 87.585 ± 0.171 | 5.3 h |

Next-event prediction: value ± half-width of the 95% bootstrap confidence interval, from a single run each.
These runs take about 3 to 5 minutes per dataset.

| Dataset | Model | RMSE<sub>Δt</sub> | Acc | sMAPE<sub>Δt</sub> | Events |
|---|---|---|---|---|---|
| Taxi | zs | 0.343 ± 0.017 | 0.469 ± 0.008 | 98.04 ± 0.93 | 14420 |
| Taxi | f | 0.458 ± 0.035 | 0.911 ± 0.004 | 85.92 ± 0.86 | 14420 |
| Taobao | zs | 0.131 ± 0.004 | 0.515 ± 0.006 | 112.89 ± 0.67 | 27955 |
| Taobao | f | 0.167 ± 0.005 | 0.595 ± 0.006 | 115.99 ± 0.73 | 27955 |

The next-event evaluation is not seeded. In repeated runs with the same checkpoint the zero-shot Taxi RMSE<sub>Δt</sub> varied between 0.343 and 0.346.
Some RMSE<sub>Δt</sub> and Acc cells of Table 2 in the paper differ from the values above in the second decimal.

## 5. Evaluation protocol

### Multi-event forecasting (Table 1)

**Context.**
- The context is the first min(2000, |train|) sequences of the train split, truncated to their first 100 events.
- In practice that is every train sequence for Taxi, Taobao and StackOverflow, and the first 2000 for Amazon and Retweet.
- The same context is used for every test sequence. It is encoded once and not updated during the rollout.

**Test sequences.**
- Every test sequence is truncated to its first 100 events (`max_num_events: 100`). This only affects the StackOverflow sequences of length 101.
- The sequence is then split into a history (the first L − N events) and a forecast horizon (the last N events).
- All test sequences are evaluated, since every sequence is longer than N.

**Rollout.** At each of the N steps:
1. FIM-PP computes the intensity from the context and the current history.
2. The thinning sampler draws candidate next-event times. It uses `dtime_max = 1.2 × max Δt in the context`, `over_sample_rate = 5` and `num_samples_boundary = 50`.
3. One candidate time t* is drawn with probability proportional to its thinning weight.
4. The mark is **sampled** from λ<sub>κ</sub>(t\*) / Σ<sub>κ'</sub> λ<sub>κ'</sub>(t\*). It is not the argmax.
5. The predicted event is **appended to the history** before the next step.

**Ensembling.**
- Each test sequence is rolled out **5 times independently** (`num_ensemble_trajectories: 5`).
- The 5 rollouts are merged into one forecast. Step k gets the mean of the 5 predicted Δt at step k and the majority-vote mark at step k (`torch.mode`; ties go to the smallest mark id).
- All metrics are computed on this merged forecast.
- Averaging over 5 rollouts is also why the spread between trials is small. A harness that scores a single sampled rollout per sequence measures a different quantity and should not be expected to match Table 1.

**Metrics.** Each metric is computed per test sequence, averaged over sequences, and reported as mean ± population std (ddof = 0) over 3 trials with seeds 0, 1, 2.
- **OTD:** computed from the cumulative predicted and true Δt, both measured from the last history event. Events are matched only within each mark. Deletion/insertion costs are C ∈ {0.05, 0.5, 1, 1.5, 2, 3, 4} and the transport cost is 1. The distance is averaged over the costs. There is no `time_range` filtering.
- **sMAPE<sub>Δt</sub>:** the mean over the N steps of 200·|Δt̂ − Δt| / (|Δt̂| + |Δt|), on raw, unnormalised inter-event times.
- **RMSE<sub>Δt</sub>:** the RMSE over the N inter-event times.
- **RMSE<sub>e</sub>:** the RMSE over marks between the predicted and true number of events of each mark.

### Next-event prediction (Table 2)

**Data.** Context and test sequences come from the EasyTPP Hugging Face datasets. The context is the first 2000 train sequences. Sequences are truncated to 100 events, and times are not shifted.

**Prediction.**
- Every event after the first of every test sequence is predicted from its true history (teacher forcing).
- The next Δt is the thinning-weighted mean of the sampled next-event times.
- The mark is the argmax of the weight-averaged λ<sub>κ</sub> / Σλ at those samples.
- This covers every event, not only the last one per sequence. The released CDiff code only scores the last event (see the footnote of Table 2).

**Metrics.**
- RMSE<sub>Δt</sub>, accuracy and sMAPE<sub>Δt</sub> are pooled over all predicted events.
- The ± is the half-width of a 95% percentile-bootstrap confidence interval over events, with 1000 resamples.

## 6. Differences between the paper text and the runs

- The captions of Tables 1 and 2 say "mean and standard deviation over 10 trials". The runs behind them used 3 trials for Table 1, and a single run with bootstrap confidence intervals for Table 2 (see above).
- Section 5.2 says that "the sequences in the test split are given as context to FIM-PP". The context is taken from the **train** split. The test sequences are only used as the histories being forecast.
- The ensembling over 5 rollouts (Section 5 above) is not described in the paper.
- Table 1 zero-shot uses epoch 36 of pretraining, and Table 2 zero-shot uses epoch 38 (see Section 3).

## 7. Re-running the fine-tuning

To fine-tune instead of downloading the fine-tuned checkpoints, run:

```bash
python scripts/hawkes/fim_benchmark.py --config scripts/hawkes/configs/fim_pp_iclr2026/finetune_table1.yaml
python scripts/hawkes/fim_benchmark.py --config scripts/hawkes/configs/fim_pp_iclr2026/finetune_table2.yaml
```

- The configs use the paper hyperparameters: NLL loss, AdamW, lr 5e-5, weight decay 1e-4, gradient clipping at 1.0, bf16 autocast.
- Table 1 fine-tunes ran for 15000 epochs, and Table 2 fine-tunes for 20000 epochs.
- An "epoch" in `fim_finetune.py` is **one optimizer step**. Each step takes a random subset of at most 2000 train sequences (truncated to 100 events) as context and one random train sequence as the target.
- The saved `best-model` is the step with the **lowest training loss**. That is the checkpoint the paper evaluated.
- Validation NLL (CDiff `dev` split or EasyTPP `validation` split) is logged every 500 steps but not used to select the checkpoint.
- The Table 1 fine-tunes started from an earlier, unreleased checkpoint of the same pretraining run (epoch 31 or earlier). Re-running from `FIM4Science/fim-pp` therefore gives numbers close to Table 1 (f), but not identical ones.
- Fine-tuning is not seeded.
