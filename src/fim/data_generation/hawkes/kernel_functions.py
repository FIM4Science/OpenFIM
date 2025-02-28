import numpy as np

from fim.leftovers_from_old_library import create_class_instance


class HawkesExpKernelFunctionSampler:
    """
    Sample parameters for functions of the form a_0 * exp(-a_1 * t).
    """

    def __init__(self, **kwargs) -> None:
        self.a_0_sampler = create_class_instance(kwargs["a_0_sampler"])
        self.a_1_sampler = create_class_instance(kwargs["a_1_sampler"])
        self.flip_sign_probability = kwargs.get("flip_sign_probability", 0)

    def __call__(self, grid_size, eps: float = 1e-3):
        """
        Evaluate the function on a grid of size grid_size.
        We choose the maximum time so that the function is less than eps after that time.

        Returns:
        t_grid: np.array
            The grid of time points.
        function_values: np.array
            The values of the function at the time points.
        """
        a_0 = self.a_0_sampler()
        a_1 = self.a_1_sampler()
        max_time = -np.log(eps / a_0) / a_1
        t_grid = np.linspace(0, max_time, grid_size)
        kernel_values = a_0 * np.exp(-a_1 * t_grid)
        if np.random.rand() < self.flip_sign_probability:
            kernel_values *= -1
        return t_grid, kernel_values


class HawkesExpSquaredKernelFunctionSampler:
    """
    Sample parameters for functions of the form a_0 * t * exp(-a_1 * t^2).
    """

    def __init__(self, **kwargs) -> None:
        self.a_0_sampler = create_class_instance(kwargs["a_0_sampler"])
        self.a_1_sampler = create_class_instance(kwargs["a_1_sampler"])
        self.flip_sign_probability = kwargs.get("flip_sign_probability", 0)

    def __call__(self, grid_size, eps: float = 1e-3):
        """
        Evaluate the function on a grid of size grid_size.
        We choose the maximum time so that the function is less than eps after that time.

        Returns:
        t_grid: np.array
            The grid of time points.
        function_values: np.array
            The values of the function at the time points.
        """
        a_0 = self.a_0_sampler()
        a_1 = self.a_1_sampler()
        max_time = np.sqrt(-np.log(eps / a_0) / a_1)  # Not quite correct but we want to avoid the Lambert W function
        t_grid = np.linspace(0, max_time, grid_size)
        kernel_values = a_0 * t_grid * np.exp(-a_1 * t_grid**2)
        if np.random.rand() < self.flip_sign_probability:
            kernel_values *= -1
        return t_grid, kernel_values


class HawkesExpShiftedKernelFunctionSampler:
    """
    Sample parameters for functions of the form a_0 * exp(-(t-a_1)/(2*a_2^2)).
    """

    def __init__(self, **kwargs) -> None:
        self.a_0_sampler = create_class_instance(kwargs["a_0_sampler"])
        self.a_1_sampler = create_class_instance(kwargs["a_1_sampler"])
        self.a_2_sampler = create_class_instance(kwargs["a_2_sampler"])
        self.flip_sign_probability = kwargs.get("flip_sign_probability", 0)

    def __call__(self, grid_size, eps: float = 1e-3):
        """
        Evaluate the function on a grid of size grid_size.
        We choose the maximum time so that the function is less than eps after that time.

        Returns:
        t_grid: np.array
            The grid of time points.
        function_values: np.array
            The values of the function at the time points.
        """
        a_0 = self.a_0_sampler()
        a_1 = self.a_1_sampler()
        a_2 = self.a_2_sampler()
        max_time = 2 * a_2**2 * np.log(a_0 / eps) + a_1
        t_grid = np.linspace(0, max_time, grid_size)
        kernel_values = a_0 * np.exp(-(t_grid - a_1) / (2 * a_2**2))
        if np.random.rand() < self.flip_sign_probability:
            kernel_values *= -1
        return t_grid, kernel_values


class HawkesExpSinKernelFunctionSampler:
    """
    Sample parameters for functions of the form a_0 * sin(a_2*t) * exp(-a_1 * t).
    """

    def __init__(self, **kwargs) -> None:
        self.a_0_sampler = create_class_instance(kwargs["a_0_sampler"])
        self.a_1_sampler = create_class_instance(kwargs["a_1_sampler"])
        self.a_2_sampler = create_class_instance(kwargs["a_2_sampler"])
        self.flip_sign_probability = kwargs.get("flip_sign_probability", 0)

    def __call__(self, grid_size, eps: float = 1e-3):
        """
        Evaluate the function on a grid of size grid_size.
        We choose the maximum time so that the function is less than eps after that time.

        Returns:
        t_grid: np.array
            The grid of time points.
        function_values: np.array
            The values of the function at the time points.
        """
        a_0 = self.a_0_sampler()
        a_1 = self.a_1_sampler()
        a_2 = self.a_2_sampler()
        max_time = -np.log(eps / a_0) / a_1  # Not quite correct but the decay gets dominated by the exp term
        t_grid = np.linspace(0, max_time, grid_size)
        kernel_values = a_0 * np.sin(a_2 * t_grid) * np.exp(-a_1 * t_grid)
        if np.random.rand() < self.flip_sign_probability:
            kernel_values *= -1
        return t_grid, kernel_values
    
class HawkesPowerLawKernelFunctionSampler:
    """
    Sample parameters for functions of the form a_0 * (t + c)^(-a_1).
    """
    def __init__(self, **kwargs) -> None:
        self.a_0_sampler = create_class_instance(kwargs["a_0_sampler"])
        self.a_1_sampler = create_class_instance(kwargs["a_1_sampler"])
        self.c_sampler = create_class_instance(kwargs["c_sampler"])  # Small constant to avoid division by zero
        self.flip_sign_probability = kwargs.get("flip_sign_probability", 0)

    def __call__(self, grid_size, eps: float = 1e-3):
        a_0 = self.a_0_sampler()
        a_1 = self.a_1_sampler()
        c = self.c_sampler()
        # Calculate max_time where function value drops below eps
        max_time = (a_0/eps)**(1/a_1) - c
        t_grid = np.linspace(0, max_time, grid_size)
        kernel_values = a_0 * (t_grid + c)**(-a_1)
        if np.random.rand() < self.flip_sign_probability:
            kernel_values *= -1
        return t_grid, kernel_values
