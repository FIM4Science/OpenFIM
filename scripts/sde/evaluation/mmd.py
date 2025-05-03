import sys
from unittest.mock import MagicMock

import numpy as np
import torch
from numpy import diag, diagonal


use_cuda = False  # We are memory bottlenecked so we don't want to use GPU and mock Cupy instead


class MockCuPy:
    random = None  # Will be set later

    def array(self, x, dtype=None):
        return np.array(x, dtype=dtype)

    def asnumpy(self, x):
        return np.array(x)

    def allclose(self, a, b, **kwargs):
        return np.allclose(a, b, **kwargs)

    def zeros(self, shape, dtype=float):
        return np.zeros(shape, dtype=dtype)

    def ones(self, shape, dtype=float):
        return np.ones(shape, dtype=dtype)

    def empty(self, shape, dtype=float):
        return np.empty(shape, dtype=dtype)

    def arange(self, *args, **kwargs):
        return np.arange(*args, **kwargs)

    def linspace(self, *args, **kwargs):
        return np.linspace(*args, **kwargs)

    def dot(self, a, b, **kwargs):
        return np.dot(a, b, **kwargs)

    def asarray(self, x, dtype=None):
        return np.asarray(x, dtype=dtype)

    def sum(self, a, axis=None, dtype=None, out=None, keepdims=False):
        return np.sum(a, axis=axis, dtype=dtype, out=out, keepdims=keepdims)

    def square(self, x):
        return np.square(x)

    def einsum(self, *args, **kwargs):
        return np.einsum(*args, **kwargs)

    def maximum(self, a, b):
        return np.maximum(a, b)

    def exp(self, x):
        return np.exp(x)

    def diff(self, a, n=1, axis=-1):
        return np.diff(a, n=n, axis=axis)

    def copy(self, a):
        return np.copy(a)

    def isscalar(self, x):
        return np.isscalar(x)

    def cumsum(self, a, axis=None):
        return np.cumsum(a, axis=axis)

    def pad(self, array, pad_width, mode="constant", **kwargs):
        return np.pad(array, pad_width, mode=mode, **kwargs)

    def stack(self, arrays, axis=0):
        return np.stack(arrays, axis=axis)

    def sqrt(self, x):
        return np.sqrt(x)

    def mean(self, a, axis=None, dtype=None, out=None, keepdims=False):
        return np.mean(a, axis=axis, dtype=dtype, out=out, keepdims=keepdims)

    # Define ndarray to be the numpy.ndarray
    ndarray = np.ndarray

    def __getattr__(self, name):
        if name == "get_array_module":
            return lambda x: np
        elif name == "_environment":

            class MockEnvironment:
                def __init__(self):
                    self.available = False
                    self._log = self.MockLog()

                class MockLog:
                    def __call__(self, *args, **kwargs):
                        pass

                    def info(self, *args, **kwargs):
                        pass

            return MockEnvironment()
        elif name in ["_core", "cuda", "cupy_backends"]:
            return MagicMock(name=name)  # Return a MagicMock for these
        raise AttributeError(f"MockCuPy has no attribute {name}")

    class Random:
        @staticmethod
        def rand(*args):
            return np.random.rand(*args)

        @staticmethod
        def randn(*args):
            return np.random.randn(*args)

        @staticmethod
        def RandomState(seed=None):
            return np.random.RandomState(seed)


if not use_cuda:
    # Create an instance of MockCuPy and set the random attribute
    mock_cupy = MockCuPy()
    mock_cupy.random = mock_cupy.Random()

    # Replace 'cupy' and related modules with our mock
    sys.modules["cupy"] = mock_cupy
    sys.modules["cupy.random"] = mock_cupy.random
    sys.modules["cupy._core"] = MagicMock()
    sys.modules["cupy.cuda"] = MagicMock()
    sys.modules["cupy_backends"] = MagicMock()


import ksig  # Follow the instruction on https://github.com/tgcsaba/KSig to install it, python: 3.10.16


# if that does not work: install cupy, clone KSIG, remove dependency on cupy, pip install numba and torch


def numpy_hash(a):
    return hash(a.tostring())


def compute_mmd(x, y, n_levels=5, kernel_cache={}):
    """
    Compute the Maximum Mean Discrepancy (MMD) between two sets of paths
    Input shapes: (n_paths, grid_size, n_dim)
    """

    assert x.shape == y.shape

    if not use_cuda and isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
        y = y.cpu().numpy()
    static_kernel = ksig.static.kernels.RBFKernel()
    sig_kernel = ksig.kernels.SignatureKernel(n_levels, static_kernel=static_kernel)

    x_hash = numpy_hash(x)

    if x_hash in kernel_cache:
        Kxx = kernel_cache[x_hash]
    else:
        Kxx = sig_kernel(x)
        kernel_cache[x_hash] = Kxx

    Kyy = sig_kernel(y)
    Kyx = sig_kernel(y, x)
    n_paths = x.shape[0]
    xx = (Kxx - diag(diagonal(Kxx))).sum() / (n_paths * (n_paths - 1))
    yy = (Kyy - diag(diagonal(Kyy))).sum() / (n_paths * (n_paths - 1))
    yx = Kyx.sum() / (n_paths * n_paths)
    return xx + yy - 2 * yx
