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
The library originates from the FIM series of papers ([References](#references)). Pretrained models that replicate the results from these publications are available on [Hugging Face](https://huggingface.co/FIM4Science). A [tutorial](https://fim4science.github.io/OpenFIM/tutorials.html) is also provided to guide users through these features.

## Installation

In order to set up the necessary environment using [uv](https://docs.astral.sh/uv/):

```bash
uv sync --python 3.12

source .venv/bin/activate

pre-commit install
pre-commit autoupdate
```

Check out the configuration under `.pre-commit-config.yaml`. The `-n, --no-verify` flag of `git commit` can be used to deactivate pre-commit hooks temporarily.

## Tutorial

Instructions on how to use the implemented models and configuration files are detailed on the [tutorial webpage](https://fim4science.github.io/OpenFIM/tutorials.html) and in [notebooks](notebooks/tutorials)

## Lamarr’s DL4SD lab

[Lamarr](https://lamarr-institute.org/)’s [Deep Learning for Scientific Discovery (DL4SD)](https://fim4science.github.io/OpenFIM/intro.html) lab is an interdisciplinary team of researchers working at the intersection of machine learning, statistical physics, and complexity science, to develop neural systems that automatically construct scientific hypotheses — articulated as mathematical models — to explain complex natural and social phenomena.

To achieve this overarching goal, we design pre-trained neural recognition models that encode classical mathematical models commonly used in the natural and social sciences. And focus on mathematical models that are simple enough to remain approximately valid across a wide range of observation scales, from microscopic to coarse-grained.

Fundamentally, these pre-trained neural recognition models enable the zero-shot inference of (the parameters defining) the mathematical equations they encode directly from data. We refer to these models as Foundation Inference Models (FIMs).

## Publications

> **Citation:** Find the bibtex entries for all publications in [publications.bib](publications.bib).

- ["Foundation Inference Models for Markov Jump Processes"](https://openreview.net/forum?id=f4v7cmm5sC) (NeurIPS 2024), David Berghaus, Kostadin Cvejoski, Patrick Seifner, César Ojeda, Ramsés J. Sánchez
- ["Zero-shot Imputation with Foundation Inference Models for Dynamical Systems"](https://openreview.net/forum?id=NPSZ7V1CCY) (ICLR 2025), Patrick Seifner, Kostadin Cvejoski, Antonia Körner, Ramsés J. Sánchez
- ["In-Context Learning of Stochastic Differential Equations with Foundation Inference Models"](https://openreview.net/forum?id=ceCJPoZOKJ) (NeurIPS 2025), Patrick Seifner, Kostadin Cvejoski, David Berghaus, César Ojeda, Ramsés J. Sánchez
- ["In-Context Learning of Temporal Point Processes with Foundation Inference Models"](https://openreview.net/forum?id=h9HwUAODFP) (ICLR 2026), David Berghaus, Patrick Seifner, Kostadin Cvejoski, César Ojeda, Ramsés J. Sánchez
- ["Foundation Inference Models for Ordinary Differential Equations"](https://arxiv.org/abs/2602.08733) (ICML 2026), Maximilian Mauel, Johannes R. Hübers, David Berghaus, Patrick Seifner, Ramsés J. Sánchez
