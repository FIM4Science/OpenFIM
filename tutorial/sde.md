# Stochastic Differential Equations: A Crash Course

This section provides a theoretical foundation for understanding Stochastic Differential Equations (SDEs) and introduces the concepts needed for the practical tutorial. This content is adapted from Appendix B of the corresponding publication {cite}`sde`.

## Mathematical Foundation

A $d$-dimensional stochastic process $x(t)$ follows an Itô stochastic differential equation (SDE) if it satisfies:

```{math}
x_i(\overline{t})=x_i(\underline{t})+\int_{\underline{t}}^{\overline{t}} f_i(x(t'),t')dt'+\sum_{j}^{m}\int_{\underline{t}}^{\overline{t}} G_{ij}(x(t'),t')dW_j(t')
```

for all $i\leq d,\ \underline{t}\leq \overline{t}$ and some vector-valued **drift function** $f:\mathbb{R}^d\times \mathbb{R}^+\to\mathbb{R}^d$ and **diffusion matrix** $G:\mathbb{R}^d\times\mathbb{R}^+\to\mathbb{R}^{d\times m}$, where $W:\mathbb{R}^+\to\mathbb{R}^m$ is a standard $m$-dimensional Wiener process. 

In differential notation, this is commonly written as:
```{math}
dx(t)=f(x(t),t)dt+G(x(t),t)dW(t)
```

## Model capabilities and assumptions

Our Foundation Inference Model (FIM) for SDEs can estimate both the drift function $f$ and diffusion function $G$ in a **zero-shot manner** directly from observed trajectory data. The model assumes **diagonal diffusion**, i.e.

```{math}
 G(x)=\text{diag}(\sqrt{g_1(x)},\dots,\sqrt{g_d(x)})
```

and therefore returns the vector field $(\sqrt{\hat{g}_1(x)},\dots,\sqrt{\hat{g}_d(x)})$.

Furthermore this model assumes **purely state-dependent drift and diffusion**!

## Bibliography
```{bibliography}
:style: alpha
:filter: docname in docnames
```