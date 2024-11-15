---
language: en
tags:
- sequence-classification
- time-series
- stochastic-processes
- markov-jump-processes
license: apache-2.0
datasets:
- custom
metrics:
- rmse
- Hellinger distance
---

# Foundation Inference Model (FIM) for Markov Jump Processes  Model Card

## Model Description

The Foundation Inference Model (`FIM`) is a neural recognition model designed for zero-shot inference of Markov Jump Processes (MJPs) in bounded state spaces. FIM processes noisy and sparse observations to estimate the transition rate matrix and initial condition of MJPs, without requiring fine-tuning on the target dataset.

FIM combines supervised learning on a synthetic dataset of MJPs with attention mechanisms, enabling robust inference for empirical processes with varying dimensionalities. It is the first generative zero-shot model for MJPs, offering broad applicability across domains such as molecular dynamics, ion channel dynamics, and discrete flashing ratchet systems.

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/cvejoski/OpenFIM)
[![arXiv](https://img.shields.io/badge/arXiv-2406.06419-B31B1B.svg)](https://arxiv.org/abs/2406.06419)

## Intended Use

- Applications:
   - Inferring dynamics of physical, chemical, and biological systems.
   - Estimating transition rates and initial conditions from noisy observations.
   - Zero-shot simulation and analysis of MJPs for:
      - Molecular dynamics simulations (e.g., alanine dipeptide conformations).
      - Protein folding models.
      - Ion channel dynamics.
      - Brownian motor systems.
- Users: Researchers in statistical physics, molecular biology, and stochastic processes.

- Limitations:
   - The model performs well only for processes with dynamics similar to its synthetic training distribution.
   - Poor estimates are likely for datasets with distributions significantly deviating from the synthetic priors (e.g., systems with power-law distributed rates).

### Installation

To install the Foundation Inference Model (FIM) from the fim library, follow these steps:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/cvejoski/OpenFIM.git
   cd OpenFIM
   ```

2. **Install the required dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Install the fim library:**

   ```bash
   pip install .
   ```

After completing these steps, you should have the `fim` library installed and ready to use.

### Description of the Input and Output Data of the Forward Step of the FIMMJP Model

#### Input Data

The input data to the forward step of the `FIMMJP` model is a dictionary containing several key-value pairs. Each key corresponds to a specific type of input data required by the model. Below is a detailed description of each key and its corresponding value:

1. **`observation_grid`**:
   - **Type**: `torch.Tensor`
   - **Shape**: `[B, P, L, 1]`
   - **Description**: This tensor represents the observation grid. `B` is the batch size, `P` is the number of paths, `L` is the length of each path, and `1` indicates a single time dimension.

2. **`observation_values`**:
   - **Type**: torch.Tensor
   - **Shape**: `[B, P, L, D]`
   - **Description**: This tensor contains the observation values. `D` is the dimensionality of the observations.

3. **`seq_lengths`**:
   - **Type**: torch.Tensor
   - **Shape**: `[B, P]`
   - **Description**: This tensor represents the sequence lengths for each path in the batch.
4. **`initial_distributions`**:
     - **Type**: torch.Tensor
     - **Shape**: `[B, N]`
     - **Description**: This tensor represents the initial distributions.

4. **Optional Keys**:
   - **`time_normalization_factors`**:
     - **Type**: torch.Tensor
     - **Shape**: `[B, 1]`
     - **Description**: This tensor represents the time normalization factors.
   - **`intensity_matrices`**:
     - **Type**: torch.Tensor
     - **Shape**: `[B, N, N]`
     - **Description**: This tensor represents the intensity matrices.

   - **`adjacency_matrices`**:
     - **Type**: torch.Tensor
     - **Shape**: `[B, N, N]`
     - **Description**: This tensor represents the adjacency matrices.

#### Output Data

The output data from the forward step of the `FIMMJP` model is a dictionary containing the following key-value pairs:

1. **`intensity_matrices`**:
   - **Type**: torch.Tensor
   - **Shape**: `[B, N, N]`
   - **Description**: This tensor represents the predicted intensity matrix for each sample in the batch. `N` is the number of states in the process.

2. **`intensity_matrices_variance`**:
   - **Type**: torch.Tensor
   - **Shape**: `[B, N, N]`
   - **Description**: This tensor represents the log variance of the predicted intensity matrix for each sample in the batch.

3. **`initial_condition`**:
   - **Type**: torch.Tensor
   - **Shape**: `[B, N]`
   - **Description**: This tensor represents the predicted initial distribution of states for each sample in the batch.

4. **`losses`** (optional):
   - **Type**: dict
   - **Description**: This dictionary contains the calculated losses if the required keys (`intensity_matrices` and `initial_distributions`) are present in the input data. The keys in this dictionary include:
     - **loss**: The total loss.
     - **loss_gauss**: The Gaussian negative log-likelihood loss.
     - **loss_initial**: The cross-entropy loss for the initial distribution.
     - **loss_missing_link**: The loss for missing links in the intensity matrix.
     - **rmse_loss**: The root mean square error loss.
     - **`beta_gauss_nll`**: The weight for the Gaussian negative log-likelihood loss.
     - **`beta_init_cross_entropy`**: The weight for the cross-entropy loss.
     - **`beta_missing_link`**: The weight for the missing link loss.
     - **`number_of_paths`**: The number of paths in the batch.

### Example Usage

Here is an example of how to use the `FIMMJP` model for inference:

```python
import torch
from transformers import AutoModel

device = "cuda" if torch.cuda.is_available() else "cpu"

# Loading the model
model = AutoModel.from_pretrained("cvejoski/FIMMJP", trust_remote_code=True)
model = model.to(device)
model.eval()

# Loading the Discrete Flashing Ratchet (DFR) dataset from Huggingface
data = load_dataset("cvejoski/mjp", download_mode="force_redownload", trust_remote_code=True, name="DFR_V=1")
data.set_format("torch")

# Create batch
inputs = {k: v.to(device) for k, v in data["train"][:1].items()}

# Perform inference
outputs = model(inputs, n_states=6)

# Process the output as needed
intensity_matrix = outputs["intensity_matrices"]
initial_distribution = outputs["initial_condition"]

print(intensity_matrix)
print(initial_distribution)
```

In this example, the input data is prepared and passed to the model's forward step. The model returns the predicted intensity matrix and initial distribution, which can then be processed as needed.

## Model Training

### Training Dataset:

- Synthetic MJPs covering state spaces ranging from 2 to 6 states, with up to 300 paths per process.
- Training spans 45,000 MJPs sampled using the Gillespie algorithm, with various grid and noise configurations.
- Noise: Includes mislabeled states (1% to 10% noise).
- Observations: Regular and irregular grids with up to 100 time points.


### Architecture:

- `Input`: K time series of noisy observations and their associated grids.
- `Encoder`: LSTM or Transformer for time-series embedding.
- `Attention`: Self-attention mechanism aggregates embeddings.
- `Output`: Transition rate matrix, variance matrix, and initial distribution.

### Loss Function:

- Supervised likelihood maximization, with regularization for missing links in the intensity matrix.

## Evaluation

The Foundation Inference Model (FIM) was evaluated on a diverse set of datasets to demonstrate its zero-shot inference capabilities for Markov Jump Processes (MJPs). The evaluation spans datasets representing different domains, such as statistical physics, molecular dynamics, and experimental biological data, testing FIM's ability to infer transition rate matrices, initial distributions, and compute physical properties like stationary distributions and relaxation times.

### Datasets

The following datasets were used to evaluate FIM:

1. Discrete Flashing Ratchet (DFR):

   - A 6-state stochastic model of a Brownian motor under a periodic potential.
   - Dataset: 5,000 paths recorded on an irregular grid of 50 time points.
   - Metrics: Transition rates, stationary distributions, and entropy production.

2. Switching Ion Channel (IonCh):

   - A 3-state model of ion flow across viral potassium channels.
   - Dataset: Experimental recordings of 5,000 paths sampled at 5kHz over one second.
   - Metrics: Mean first-passage times, stationary distributions.

3. Alanine Dipeptide (ADP):

   - A 6-state molecular dynamics model describing dihedral angles of alanine dipeptide.
   - Dataset: 1 microsecond simulation of atom trajectories, mapped to coarse-grained states.
   - Metrics: Relaxation times, stationary distributions.

4. Simple Protein Folding (PFold):

   - A 2-state model describing folding and unfolding rates of proteins.
   - Dataset: Simulated transitions between folded and unfolded states.
   - Metrics: Transition rates, mean first-passage times.


#### Summary

The Foundation Inference Model (FIM) represents a groundbreaking approach to zero-shot inference for Markov Jump Processes (MJPs). FIM enables accurate estimation of a variety of properties, including stationary distributions, relaxation times, mean first-passage times, time-dependent moments, and thermodynamic quantities (e.g., entropy production), all from noisy and discretely observed MJPs with state spaces of varying dimensionalities. Importantly, FIM operates in a zero-shot mode, requiring no additional fine-tuning or retraining on target datasets.

To the best of our knowledge, FIM is the first zero-shot generative model for MJPs, showcasing a versatile and powerful methodology for a wide range of physical, chemical, and biological systems. Future directions for FIM include extending its applicability to Birth and Death processes and incorporating more complex prior distributions for transition rates to enhance its generalization capabilities.

## Limitations

While FIM has demonstrated strong performance on synthetic datasets, its methodology relies heavily on the distribution of these synthetic data. As a result, the model's effectiveness diminishes when evaluated on empirical datasets that significantly deviate from the synthetic distribution. For instance, as shown in Figure 4 (right), FIM's performance degrades rapidly for cases where the ratio between the largest and smallest transition rates exceeds three orders of magnitude. Such scenarios fall outside the range of FIM's prior Beta distributions and present challenges for accurate inference.

Additionally, the dynamics of MJPs underlying systems with long-lived, metastable states depend heavily on the shape of the energy landscape defining the state space. Transition rates in these systems are characterized by the depth of energy traps and can follow distributions not represented in FIM's training prior, such as power-law distributions (e.g., in glassy systems). These distributions lie outside the synthetic ensemble used for training FIM, limiting its ability to generalize to such cases.

To address these limitations, future work will explore training FIM on synthetic MJPs with more complex transition rate distributions, such as those arising from systems with exponentially distributed energy traps or power-law-distributed rates, to better handle a broader range of real-world scenarios.
## License

The model is licensed under the Apache 2.0 License.

## Citation

If you use this model in your research, please cite:

```
@article{berghaus2024foundation,
  title={Foundation Inference Models for Markov Jump Processes},
  author={Berghaus, David and Cvejoski, Kostadin and Seifner, Patrick and Ojeda, Cesar and Sanchez, Ramses J},
  journal={arXiv preprint arXiv:2406.06419},
  year={2024}
}
```

## Contact

For questions or issues, please contact Kostadin Cvejoski at cvejoski@gmail.com.
