# Markov Jump Processes: A Crash Course

This section is an adapted version of Appendix A in {cite}`fim_mjp`. We introduce
the notion of Markov jump processes and several important definitions and notation to work with them.

## What are MJPs?

Markov Jump Processes (MJPs) are stochastic models used to describe systems that transition between
states at random times. These processes are characterized by the Markov property where the future
state depends only on the current state, not on the sequence of events that preceded it.

A continuous-time MJP $X(t)$ has right-continuous, piecewise-constant paths and takes values in a countable state space $\mathcal{X}$ over a time interval $[0, T]$. The instantaneous probability rate of transitioning from state $x'$ to $x$ is defined as
```{math}
f(x|x', t) = \lim_{\Delta t \rightarrow 0} \frac{1}{\Delta t} p_{\tiny\text{MJP}}(x, t+\Delta t| x', t),
```
where $p_{\tiny\text{MJP}}(x, t| x', t')$ denotes the transition probability.

## Master equation

The evolution of the state probabilities $p_{\tiny\text{MJP}}(x, t)$ is governed by the master equation
```{math}
    \frac{d p_{\tiny\text{MJP}} (x, t)}{dt}  =  \sum_{x' \neq x} \Big( f(x|x') p_{\tiny\text{MJP}}(x', t) - f(x'| x)p_{\tiny\text{MJP}}(x, t) \Big).
```

For homogeneous MJPs with time-independent transition rates, the master equation in matrix form is
```{math}
\frac{d p_{\tiny\text{MJP}} (x, t)}{dt}(t) = \mathbf{p_{\tiny\text{MJP}}}(t) \cdot \mathbf{F},
```
with the solution given by the matrix exponential
```{math}
\mathbf{p_{\tiny\text{MJP}}}(t) = \mathbf{p_{\tiny\text{MJP}}}(0) \cdot \exp(\mathbf{F}t).
```
## Stationary Distribution
The stationary distribution $\mathbf{p^*_{\tiny\text{MJP}}}$ of a homogeneous MJP is a probability distribution over the state space $\mathcal{X}$ that satisfies the condition $\mathbf{p^*_{\tiny\text{MJP}}} \cdot \mathbf{F} = \mathbf{0}$. This implies that the stationary distribution is a left eigenvector of the rate matrix corresponding to the eigenvalue 0.

## Relaxation Times
The relaxation time of a homogeneous MJP is determined by its non-zero eigenvalues $\lambda_2, \lambda_3, \ldots, \lambda_{|\mathcal{X}|}$. These eigenvalues define the time scales of the process: $|\text{Re}(\lambda_2)|^{-1}, |\text{Re}(\lambda_3)|^{-1}, \ldots, |\text{Re}(\lambda_{|\mathcal{X}|})|^{-1}$. These time scales are indicative of the exponential rates of decay toward the stationary distribution. The relaxation time, which is the longest of these time scales, dominates the long-term convergence behavior. If the eigenvalue corresponding to the relaxation time has a non-zero imaginary part, then this means that the system does not converge into a fixed stationary distribution but that it instead ends in a periodic oscillation.

## Mean First-Passage Times (MFPT)
For an MJP starting in a state $i \in \mathcal{X}$, the first-passage time to another state $j \in \mathcal{X}$ is defined as the earliest time $t$ at which the MJP reaches state $j$, given it started in state $i$. The mean first-passage time (MFPT) $\tau_{ij}$ is the expected value of this time. For a finite state, time-homogeneous MJP, the MFPTs can be determined by solving a series of linear equations for each state $j$, distinct from $i$, with the initial condition that $\tau_{ii} = 0$
```{math}
    \begin{cases}
\tau_{ii} = 0 & \\
1 + \sum_{k} \mathbf{F}_{ik} \tau_{kj} = 0, & j \neq i
\end{cases}
```
