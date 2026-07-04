# The FIM Tutorial

This tutorial gives a guided introduction to the [FIM library](https://github.com/FIM4Science/OpenFIM) and the models trained for the "Foundation Models for Inference" series of papers.
The basic usage of this library is discussed together with a detailed documentation of the user interface, which consists of configuration files in the yaml format.

Afterwards the individual papers are shortly discussed by introducing motivating problems from different domains, which then are solved with the help
of our trained models.

These currently include FIMs for:
- [Markov Jump Processes](../03_mjp/mjp_crashcourse.md)
- [Point Processes](../04_pp/pp_crashcourse.md)
- [Stochastic Differential Equations](../05_sde/sde_crashcourse.md)
- [Ordinary Differential Equations](../06_ode/ode_tutorial.ipynb)
- [Zero-shot Time Series Imputation](../07_imputation/imputation.ipynb)

Note that if you are interested in training your own model [the installation section](../02_library/overview.md) contains a step by step
description on how to build you training configuration file. If you are primarily interested in using our models you can safely skip to the chapter
on the desired problem class after installing FIM.
