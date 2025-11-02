```{warning}
OpenFim is currently being updated, therefore some of the code will temporarily not work, due to some internal updates to some of the core blocks of the models! Previous versions of the tutorial can be found on github and the old MJP model can still be downloaded from huggingface.
```

# Stochastic Differential Equations: A Crash Course

This section is an adapted version of Appendix A of the corresponding publication {cite}`sde`.

A $d$ dimensional stochastic process $x(t)$ follows an Ito stochastic differential equation (SDE), if it satisfies

```{math}
x_i(\overline{t})=x_i(\overline{t})+\int_{\underline{t}}^{\overline{t}} f_i(x(t'),t')dt'+\sum_{j}^{m}\int_{\underline{t}}^{\overline{t}} G_{ij}(x(t'),t')dW_j(t')
```

for all $i\leq d,\ \underline{t}\leq \overline{t}$ and some vector valued *drift function* $f:\mathbb{R}^d\times \mathbb{R}^+\to\mathbb{R}^d$ and *diffusion function* $G:\mathbb{R}^d\times\mathbb{R}^+\to\mathbb{R}^{m\times d}$, where $W:\mathbb{R}^+\to\mathbb{R}$ is a standard $m$-dimensional Wiener process. Such a process can be denoted in differential notation:
```{math}
dx(t)=f(x(t),t)dt+G(x(t),t)dW(t)
```

Our model estimates both $f$ and $G$ in a zero-shot manner, where $G$ is returned in the following form:
```{math}
 G(x)=\text{diag}(\sqrt{g_1(x)},\dots,\sqrt{g_d(x)})
```

## A few notes
1. The model and the tutorial assume, that both the drift and the diffusion are only state-dependent and not time dependent.
2. Some authors return a different scaling of $G$: 
```{math}
\text{diag}(g_1(x),\dots,g_d(x)).
```
Therefore one should be careful when comparing results from different models/ authors!

The next section of this tutorial will demonstrate how one can use our model
in practice to produce predictions for the underlying SDE of some given data.

![alt text](sde1.png)