# The FIM tutorial

This tutorial gives a guided introduction to the FIM library and the models trained for the "Foundation Models for Inference" series of papers.
The basic usage of this library is discussed together with a detailed documentation of the user interface, which consists of configuration files in the yaml format.

Afterwards the individual papers are shortly discussed by introducing motivating problems from different domains, which then are solved with the help
of our trained models.

Note that if you are interested in training you own model [FIM_installation_and_settings](FIM_installation_and_settings.md) contains a step by step
description on how to build you training configuration file. If you are primarily interested in using our models you can safely skip to the chapter
on the desired problem class after reading [FIM_installation_and_settings](FIM_installation_and_settings.md).

## About us

Lamarr's Deep Learning for Scientific Discovery (DL4SD) group is an interdisciplinary team of researchers working at the intersection of machine learning, statistical physics, and complexity science, to develop neural systems that automatically construct scientific hypotheses — articulated as mathematical models — to explain complex natural and social phenomena.

To achieve this overarching goal, we design pre-trained neural recognition models that encode classical mathematical models commonly used in the natural and social sciences. And focus on mathematical models that are simple enough to remain approximately valid across a wide range of observation scales, from microscopic to coarse-grained.

## About Foundation Inference Models (FIMs)

Neural networks encode information through data processing. We therefore leverage simulations of the mathematical models we seek to encode (that is, their approximate solutions), to connect the data representing these solutions with the corresponding equations. Once encoded into our neural networks, (the representations of) these mathematical models can be compared and combined by both humans and neural reasoners trained to construct novel scientific hypotheses directly from data.

Fundamentally, these pre-trained neural recognition models enable the zero-shot inference of (the parameters defining) the mathematical equations they encode directly from data. We refer to these models as Foundation Inference Models (FIMs).

## Publications

Previous work of our group consists of the following papers

1. Zero-Shot Imputation with Foundation Inference Models for Dynamical Systems {cite}`fim_dynamical_systems`
2. Foundation Inference Models for Markov Jump Processes {cite}`fim_mjp`


## Bibliography
```{bibliography}
:style: alpha
:filter: docname in docnames
```
