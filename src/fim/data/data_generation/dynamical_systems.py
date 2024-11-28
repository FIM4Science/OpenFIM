from abc import ABC, abstractmethod
from itertools import combinations

import torch
from torch import Tensor


class DynamicalSystem(ABC):
    """
    Abstract class to define dynamical systems for data generation
    """

    name_str: str
    num_realizations: int
    redo: bool
    data_bulk_name: str
    state_dim: int

    def __init__(self, config):
        self.config = config
        self.num_realizations = config.get("num_realizations")
        self.redo = config.get("redo")
        self.data_bulk_name = config.get("data_bulk_name")

        self.drift_params = config.get("drift_params")
        self.diffusion_params = config.get("diffusion_params")
        self.initial_state = config.get("initial_state")

    @abstractmethod
    def drift(self, states, time, params) -> Tensor:
        """Defines the drift component of the SDE."""
        pass

    @abstractmethod
    def diffusion(self, states, time, params) -> Tensor:
        """Defines the diffusion component of the SDE."""
        pass

    @abstractmethod
    def sample_drift_params(self, num_paths) -> Tensor:
        """Samples drift parameters specific to the dynamical system."""
        pass

    @abstractmethod
    def sample_diffusion_params(self, num_paths) -> Tensor:
        """Samples diffusion parameters specific to the dynamical system."""
        pass

    @abstractmethod
    def sample_initial_states(self, num_paths) -> Tensor:
        """Defines the initial states for the system."""
        pass

    def sample_diffusion_params_generic(self, num_paths):
        # Initialize an empty list to store the sampled parameters
        samples_list = []

        # Iterate through each parameter in diffusion_params
        for key, config in self.diffusion_params.items():
            # Check the distribution type for each parameter
            if config["distribution"] == "uniform":
                param_min = config["min"]
                param_max = config["max"]
                param_dist = torch.distributions.uniform.Uniform(param_min, param_max)
                param_samples = param_dist.sample((num_paths,))

            elif config["distribution"] == "fix":
                # If fixed, fill with the fixed value
                param_samples = torch.full((num_paths,), config.get("fix_value", 0.0))

            else:
                # Raise an error for unsupported distribution types
                raise ValueError(f"Unsupported distribution type '{config['distribution']}' for parameter '{key}'")

            # Append the sampled tensor to the list
            samples_list.append(param_samples)

        # Stack all samples along the second dimension to create the final tensor
        return torch.stack(samples_list, dim=1)

    def sample_initial_states_generic(self, num_paths):
        if self.initial_state["distribution"] == "normal":
            mean = self.initial_state["mean"]
            std_dev = self.initial_state["std_dev"]
            initial_states = torch.normal(mean, std_dev, size=(num_paths, self.state_dim))
        elif self.initial_state["distribution"] == "fix":
            initial_states = torch.Tensor(self.initial_state["fix_value"])
            initial_states = initial_states.repeat((num_paths, 1))
        elif self.initial_state["distribution"] == "uniform":
            _min = self.initial_state["min"]
            _max = self.initial_state["max"]
            initial_states = _min + torch.rand(size=(num_paths, self.state_dim)) * (_max - _min)

        if self.initial_state["activation"] == "sigmoid":
            initial_states = torch.sigmoid(initial_states)

        return initial_states


class Lorenz63System(DynamicalSystem):
    """ """

    name_str: str = "LorenzSystem63"
    state_dim: int = 3

    def __init__(self, config: dict):
        super().__init__(config)

    def drift(self, states, time, params):
        sigma, beta, rho = params[:, 0], params[:, 1], params[:, 2]
        x, y, z = states[:, 0], states[:, 1], states[:, 2]
        dxdt = sigma * (y - x)
        dydt = x * (rho - z) - y
        dzdt = x * y - beta * z
        return torch.stack([dxdt, dydt, dzdt], dim=1)

    def diffusion(self, states, time, params):
        return params.to(states.device)

    def sample_drift_params(self, num_paths):
        if self.drift_params["sigma"]["distribution"] == "uniform":
            sigma_dist = torch.distributions.uniform.Uniform(self.drift_params["sigma"]["min"], self.drift_params["sigma"]["max"])
            sigma_samples = sigma_dist.sample((num_paths,))
        elif self.drift_params["sigma"]["distribution"] == "fix":
            sigma_samples = torch.full(size=(num_paths,), fill_value=self.drift_params["sigma"]["fix_value"])

        if self.drift_params["beta"]["distribution"] == "uniform":
            beta_dist = torch.distributions.uniform.Uniform(self.drift_params["beta"]["min"], self.drift_params["beta"]["max"])
            beta_samples = beta_dist.sample((num_paths,))
        elif self.drift_params["beta"]["distribution"] == "fix":
            beta_samples = torch.full(size=(num_paths,), fill_value=self.drift_params["beta"]["fix_value"])

        if self.drift_params["rho"]["distribution"] == "uniform":
            rho_dist = torch.distributions.uniform.Uniform(self.drift_params["rho"]["min"], self.drift_params["rho"]["max"])
            rho_samples = rho_dist.sample((num_paths,))
        elif self.drift_params["rho"]["distribution"] == "fix":
            rho_samples = torch.full(size=(num_paths,), fill_value=self.drift_params["rho"]["fix_value"])

        return torch.stack([sigma_samples, beta_samples, rho_samples], dim=1)

    def sample_diffusion_params(self, num_paths):
        # Constant diffusion parameter across states
        constant_value = self.diffusion_params["constant_value"]
        dimensions = self.diffusion_params["dimensions"]
        return torch.full((num_paths, dimensions), constant_value)

    def sample_initial_states(self, num_paths):
        if self.initial_state["distribution"] == "normal":
            mean = self.initial_state["mean"]
            std_dev = self.initial_state["std_dev"]
            dimensions = self.state_dim
            initial_states = torch.normal(mean, std_dev, size=(num_paths, dimensions))
        elif self.initial_state["distribution"] == "fix":
            initial_states = torch.Tensor(self.initial_state["fix_value"])
            initial_states = initial_states.repeat((num_paths, 1))

        if self.initial_state["activation"] == "sigmoid":
            initial_states = torch.sigmoid(initial_states)

        return initial_states


class HopfBifurcation(DynamicalSystem):
    name_str: str = "HopfBifurcation"
    state_dim: int = 2

    def __init__(self, config: dict):
        super().__init__(config)

    def drift(self, states, time, params):
        sigma, beta, rho = params[:, 0], params[:, 1], params[:, 2]
        x, y = states[:, 0], states[:, 1]

        dxdt = sigma * x + y - rho * x * (x**2 + y**2)
        dydt = -x + beta * y - rho * y * (x**2 + y**2)

        return torch.stack([dxdt, dydt], dim=1)

    def diffusion(self, states, time, params):
        return params.to(states.device)

    def sample_drift_params(self, num_paths):
        if self.drift_params["sigma"]["distribution"] == "uniform":
            sigma_dist = torch.distributions.uniform.Uniform(self.drift_params["sigma"]["min"], self.drift_params["sigma"]["max"])
            sigma_samples = sigma_dist.sample((num_paths,))
        elif self.drift_params["sigma"]["distribution"] == "fix":
            sigma_samples = torch.full(size=(num_paths,), fill_value=self.drift_params["sigma"]["fix_value"])

        if self.drift_params["beta"]["distribution"] == "uniform":
            beta_dist = torch.distributions.uniform.Uniform(self.drift_params["beta"]["min"], self.drift_params["beta"]["max"])
            beta_samples = beta_dist.sample((num_paths,))
        elif self.drift_params["beta"]["distribution"] == "fix":
            beta_samples = torch.full(size=(num_paths,), fill_value=self.drift_params["beta"]["fix_value"])

        if self.drift_params["rho"]["distribution"] == "uniform":
            rho_dist = torch.distributions.uniform.Uniform(self.drift_params["rho"]["min"], self.drift_params["rho"]["max"])
            rho_samples = rho_dist.sample((num_paths,))
        elif self.drift_params["rho"]["distribution"] == "fix":
            rho_samples = torch.full(size=(num_paths,), fill_value=self.drift_params["rho"]["fix_value"])

        return torch.stack([sigma_samples, beta_samples, rho_samples], dim=1)

    def sample_diffusion_params(self, num_paths):
        # Constant diffusion parameter across states
        return self.sample_diffusion_params_generic(num_paths)

    def sample_initial_states(self, num_paths):
        return self.sample_initial_states_generic(num_paths)


class DampedCubicOscillatorSystem(DynamicalSystem):
    name_str: str = "DampedCubicOscillatorSystem"
    state_dim: int = 2

    def __init__(self, config: dict):
        super().__init__(config)

    def drift(self, states, time, params):
        damping, alpha, beta = params[:, 0], params[:, 1], params[:, 2]
        x1, x2 = states[:, 0], states[:, 1]
        dx1dt = -(damping * x1**3 - alpha * x2**3)
        dx2dt = -(beta * x1**3 + damping * x2**3)
        return torch.stack([dx1dt, dx2dt], dim=1)

    def diffusion(self, states, time, params):
        return params.to(states.device)

    def sample_drift_params(self, num_paths):
        # Define distributions for parameters
        if self.drift_params["damping"]["distribution"] == "uniform":
            damping_dist = torch.distributions.uniform.Uniform(self.drift_params["damping"]["min"], self.drift_params["damping"]["max"])
            damping_samples = damping_dist.sample((num_paths,))
        elif self.drift_params["damping"]["distribution"] == "fix":
            damping_samples = torch.full(size=(num_paths,), fill_value=self.drift_params["damping"]["fix_value"])

        if self.drift_params["alpha"]["distribution"] == "uniform":
            alpha_dist = torch.distributions.uniform.Uniform(self.drift_params["alpha"]["min"], self.drift_params["alpha"]["max"])
            alpha_samples = alpha_dist.sample((num_paths,))
        elif self.drift_params["alpha"]["distribution"] == "fix":
            alpha_samples = torch.full(size=(num_paths,), fill_value=self.drift_params["alpha"]["fix_value"])

        if self.drift_params["beta"]["distribution"] == "uniform":
            beta_dist = torch.distributions.uniform.Uniform(self.drift_params["beta"]["min"], self.drift_params["beta"]["max"])
            beta_samples = beta_dist.sample((num_paths,))
        elif self.drift_params["beta"]["distribution"] == "fix":
            beta_samples = torch.full(size=(num_paths,), fill_value=self.drift_params["beta"]["fix_value"])

        return torch.stack([damping_samples, alpha_samples, beta_samples], dim=1)

    def sample_diffusion_params(self, num_paths):
        return self.sample_diffusion_params_generic(num_paths)

    def sample_initial_states(self, num_paths):
        # Initial conditions set to 1.0 for each state variable
        return self.sample_initial_states_generic(num_paths)


class DampedLinearOscillatorSystem(DynamicalSystem):
    name_str: str = "DampedLinearOscillatorSystem"
    state_dim: int = 2

    def __init__(self, config: dict):
        super().__init__(config)

    def drift(self, states, time, params):
        damping, alpha, beta = params[:, 0], params[:, 1], params[:, 2]
        x1, x2 = states[:, 0], states[:, 1]
        dx1dt = -(damping * x1 - alpha * x2)
        dx2dt = -(beta * x1 + damping * x2)
        return torch.stack([dx1dt, dx2dt], dim=1)

    def diffusion(self, states, time, params):
        return params.to(states.device)

    def sample_drift_params(self, num_paths):
        # Define distributions for parameters
        if self.drift_params["damping"]["distribution"] == "uniform":
            damping_dist = torch.distributions.uniform.Uniform(self.drift_params["damping"]["min"], self.drift_params["damping"]["max"])
            damping_samples = damping_dist.sample((num_paths,))
        elif self.drift_params["damping"]["distribution"] == "fix":
            damping_samples = torch.full(size=(num_paths,), fill_value=self.drift_params["damping"]["fix_value"])

        if self.drift_params["alpha"]["distribution"] == "uniform":
            alpha_dist = torch.distributions.uniform.Uniform(self.drift_params["alpha"]["min"], self.drift_params["alpha"]["max"])
            alpha_samples = alpha_dist.sample((num_paths,))
        elif self.drift_params["alpha"]["distribution"] == "fix":
            alpha_samples = torch.full(size=(num_paths,), fill_value=self.drift_params["alpha"]["fix_value"])

        if self.drift_params["beta"]["distribution"] == "uniform":
            beta_dist = torch.distributions.uniform.Uniform(self.drift_params["beta"]["min"], self.drift_params["beta"]["max"])
            beta_samples = beta_dist.sample((num_paths,))
        elif self.drift_params["beta"]["distribution"] == "fix":
            beta_samples = torch.full(size=(num_paths,), fill_value=self.drift_params["beta"]["fix_value"])

        return torch.stack([damping_samples, alpha_samples, beta_samples], dim=1)

    def sample_diffusion_params(self, num_paths):
        return self.sample_diffusion_params_generic(num_paths)

    def sample_initial_states(self, num_paths):
        # Initial conditions set to 1.0 for each state variable
        return self.sample_initial_states_generic(num_paths)


class DuffingOscillator(DynamicalSystem):
    name_str: str = "DuffingOscillator"
    state_dim: int = 2

    def __init__(self, config: dict):
        super().__init__(config)

    def drift(self, states, time, params):
        alpha, beta, gamma = params[:, 0], params[:, 1], params[:, 2]
        x1, x2 = states[:, 0], states[:, 1]
        dx1dt = alpha * x2
        dx2dt = -(x1**3 - beta * x1 + gamma * x2)
        return torch.stack([dx1dt, dx2dt], dim=1)

    def diffusion(self, states, time, params):
        return params.to(states.device)

    def sample_drift_params(self, num_paths):
        # Define distributions for parameters
        if self.drift_params["alpha"]["distribution"] == "uniform":
            alpha_dist = torch.distributions.uniform.Uniform(self.drift_params["alpha"]["min"], self.drift_params["alpha"]["max"])
            alpha_samples = alpha_dist.sample((num_paths,))
        elif self.drift_params["alpha"]["distribution"] == "fix":
            alpha_samples = torch.full(size=(num_paths,), fill_value=self.drift_params["alpha"]["fix_value"])

        if self.drift_params["beta"]["distribution"] == "uniform":
            beta_dist = torch.distributions.uniform.Uniform(self.drift_params["beta"]["min"], self.drift_params["beta"]["max"])
            beta_samples = beta_dist.sample((num_paths,))
        elif self.drift_params["beta"]["distribution"] == "fix":
            beta_samples = torch.full(size=(num_paths,), fill_value=self.drift_params["beta"]["fix_value"])

        if self.drift_params["gamma"]["distribution"] == "uniform":
            gamma_dist = torch.distributions.uniform.Uniform(self.drift_params["gamma"]["min"], self.drift_params["gamma"]["max"])
            gamma_samples = gamma_dist.sample((num_paths,))
        elif self.drift_params["gamma"]["distribution"] == "fix":
            gamma_samples = torch.full(size=(num_paths,), fill_value=self.drift_params["gamma"]["fix_value"])

        # Sample parameters for all paths
        return torch.stack([alpha_samples, beta_samples, gamma_samples], dim=1)

    def sample_diffusion_params(self, num_paths):
        return self.sample_diffusion_params_generic(num_paths)

    def sample_initial_states(self, num_paths):
        return self.sample_initial_states_generic(num_paths)


class SelkovGlycosis(DynamicalSystem):
    name_str: str = "SelkovGlycosis"
    state_dim: int = 2

    def __init__(self, config: dict):
        super().__init__(config)

    def drift(self, states, time, params):
        alpha, beta, gamma = params[:, 0], params[:, 1], params[:, 2]
        x1, x2 = states[:, 0], states[:, 1]
        dx1dt = -(x1 - alpha * x2 - (x1**2) * x2)
        dx2dt = gamma - beta * x1 - (x1**2) * x2
        return torch.stack([dx1dt, dx2dt], dim=1)

    def diffusion(self, states, time, params):
        return params.to(states.device)

    def sample_drift_params(self, num_paths):
        # Define distributions for parameters
        if self.drift_params["alpha"]["distribution"] == "uniform":
            alpha_dist = torch.distributions.uniform.Uniform(self.drift_params["alpha"]["min"], self.drift_params["alpha"]["max"])
            alpha_samples = alpha_dist.sample((num_paths,))
        elif self.drift_params["alpha"]["distribution"] == "fix":
            alpha_samples = torch.full(size=(num_paths,), fill_value=self.drift_params["alpha"]["fix_value"])

        if self.drift_params["beta"]["distribution"] == "uniform":
            beta_dist = torch.distributions.uniform.Uniform(self.drift_params["beta"]["min"], self.drift_params["beta"]["max"])
            beta_samples = beta_dist.sample((num_paths,))
        elif self.drift_params["beta"]["distribution"] == "fix":
            beta_samples = torch.full(size=(num_paths,), fill_value=self.drift_params["beta"]["fix_value"])

        if self.drift_params["gamma"]["distribution"] == "uniform":
            gamma_dist = torch.distributions.uniform.Uniform(self.drift_params["gamma"]["min"], self.drift_params["gamma"]["max"])
            gamma_samples = gamma_dist.sample((num_paths,))
        elif self.drift_params["gamma"]["distribution"] == "fix":
            gamma_samples = torch.full(size=(num_paths,), fill_value=self.drift_params["gamma"]["fix_value"])

        return torch.stack([alpha_samples, beta_samples, gamma_samples], dim=1)

    def sample_diffusion_params(self, num_paths):
        # Constant diffusion parameter across states
        return self.sample_diffusion_params_generic(num_paths)

    def sample_initial_states(self, num_paths):
        # Initial conditions set to 1.0 for each state variable
        return self.sample_initial_states_generic(num_paths)


class DoubleWellOneDimension(DynamicalSystem):
    name_str: str = "DoubleWellOneDimension"
    state_dim: int = 1

    def __init__(self, config):
        super().__init__(config)

    def drift(self, states, time, params) -> Tensor:
        alpha, beta = params[:, 0], params[:, 1]
        x1 = states
        dx1dt = alpha[:, None] * x1 - beta[:, None] * x1**3
        return dx1dt

    def diffusion(self, states, time, params) -> Tensor:
        x1 = states
        g1, g2 = params[:, 0], params[:, 1]
        dx = g1[:, None] - g2[:, None] * (x1**2)
        dx = torch.clip(dx, min=0)
        dx = torch.sqrt(dx)
        return dx

    def sample_drift_params(self, num_paths) -> Tensor:
        # Define distributions for parameters
        if self.drift_params["alpha"]["distribution"] == "uniform":
            alpha_dist = torch.distributions.uniform.Uniform(self.drift_params["alpha"]["min"], self.drift_params["alpha"]["max"])
            alpha_samples = alpha_dist.sample((num_paths,))
        elif self.drift_params["alpha"]["distribution"] == "fix":
            alpha_samples = torch.full(size=(num_paths,), fill_value=self.drift_params["alpha"]["fix_value"])

        if self.drift_params["beta"]["distribution"] == "uniform":
            beta_dist = torch.distributions.uniform.Uniform(self.drift_params["beta"]["min"], self.drift_params["beta"]["max"])
            beta_samples = beta_dist.sample((num_paths,))
        elif self.drift_params["beta"]["distribution"] == "fix":
            beta_samples = torch.full(size=(num_paths,), fill_value=self.drift_params["beta"]["fix_value"])
        return torch.stack([alpha_samples, beta_samples], dim=1)

    def sample_diffusion_params(self, num_paths) -> Tensor:
        return self.sample_diffusion_params_generic(num_paths)

    def sample_initial_states(self, num_paths) -> Tensor:
        return self.sample_initial_states_generic(num_paths)


class Degree2Polynomial(DynamicalSystem):
    """
    Drift and diffusion are polynomials of (up to) degree 2. Coefficients are sampled from specified distribution.
    Drift value is simply the polynomial. Diffusion value is made positive (or zero) and its square root is returned.
    """

    name_str: str = "Degree2Polynomial"

    def __init__(self, config):
        super().__init__(config)

        # number of coefficients in each coefficient group
        self.state_dim: int = config.get("state_dim", 1)
        self.coeffs_count = {
            "constant": 1,
            "degree_1": self.state_dim,
            "degree_2_squared": self.state_dim,
            "degree_2_mixed": self.state_dim * (self.state_dim - 1) // 2,
        }

        # make diffusion value positive, before taking square rood
        self.enforce_positivity: str = config.get("enforce_positivity", "clip")
        assert self.enforce_positivity in ["clip", "exp", "abs"]

        # optionally print sampled equations
        self.show_equation: bool = config.get("show_equation", False)
        self.precision: int = config.get("precision", 3)

    def sample_polynomial_coeffs(self, num_realizations: int, sample_params: dict) -> Tensor:
        """
        Sample state_dim (up to) degree 2 polynomials with state_dim variables. Coefficients are sampled in 4 groups:
        Constant term, degree 1 (linear), degree 2 wquared (i.e. x_i^2) and degree 2 mixed (i.e. x_i * x_j)

        Args:
            num_realizations (int): Number of (polynomial) vector fields to sample.
            sample_params (dict): Define distributions to sample coefficients from.

        Returns:
            params (Tensor): Contains sampled coefficients. Shape: [num_realizations, state_dim, total_coeffs_count]
        """
        coeffs_samples = {}

        for coeff_label, coeff_count in self.coeffs_count.items():
            # distributions for coefficient group
            if sample_params[coeff_label]["distribution"] == "uniform":
                coeff_dist = torch.distributions.uniform.Uniform(sample_params[coeff_label]["min"], sample_params[coeff_label]["max"])
                coeff_samples = coeff_dist.sample((num_realizations, self.state_dim, coeff_count))

            elif sample_params[coeff_label]["distribution"] == "normal":
                coeff_dist = torch.distributions.normal.Normal(0, sample_params[coeff_label]["std"])
                coeff_samples = coeff_dist.sample((num_realizations, self.state_dim, coeff_count))

            elif sample_params[coeff_label]["distribution"] == "fix":
                coeff_samples = torch.full(
                    size=(num_realizations, self.state_dim, coeff_count), fill_value=sample_params[coeff_label]["fix_value"]
                )

            # drop some coefficients at random; probability set for all coefficients
            if (p := sample_params[coeff_label].get("bernoulli_survival_rate")) is not None:
                survival_dist = torch.distributions.bernoulli.Bernoulli(probs=p)
                survival_mask = survival_dist.sample((num_realizations, self.state_dim, coeff_count))

                coeff_samples = coeff_samples * survival_mask

            coeffs_samples.update({coeff_label: coeff_samples})

        # Combine sampled coefficients to single tensor
        params = torch.concatenate(list(coeffs_samples.values()), dim=-1)  # [num_realizations, state_dim, total_coeffs_count]

        # sample scale parameter
        if "scale" in sample_params.keys():
            sample_per_dimension: bool = sample_params["scale"].get("sample_per_dimension", False)
            scale_size = (num_realizations, self.state_dim if sample_per_dimension is True else 1, 1)

            if sample_params["scale"]["distribution"] == "uniform":
                scale_dist = torch.distributions.uniform.Uniform(sample_params["scale"]["min"], sample_params["scale"]["max"])
                scale = scale_dist.sample(scale_size)
            elif sample_params["scale"]["distribution"] == "fix":
                scale = torch.full(size=scale_size, fill_value=sample_params["scale"]["fix_value"])

        else:
            scale = torch.ones(num_realizations, 1, 1)

        # same scale for all dimensions
        scale = scale.expand(-1, self.state_dim, -1)  # [num_realizations, state_dim, 1]
        params = torch.concatenate([params, scale], dim=-1)

        return params

    def extract_coefficients(self, params: Tensor) -> tuple[Tensor]:
        """
        Extract coefficients per monomial groups from output of self.sample_diffusion_params.

        Args:
            params (Tensor): Sampled coefficients, output from self.sample_diffusion_params. Shape: [num_realizations, state_dim, total_coeffs_count]

        Returns:
            coefficients of monomial groups (tuple[Tensor]): Shapes: [num_realizations, state_dim, coeff_count]
        """
        const_coeff = params[..., :1]  # [..., 1]
        deg_1_coeffs = params[..., 1 : 1 + self.state_dim]  # [..., self.state_dim]
        deg_2_squared_coeffs = params[..., 1 + self.state_dim : 1 + 2 * self.state_dim]  # [..., self.state_dim]

        assert deg_1_coeffs.shape[-1] == self.coeffs_count["degree_1"]
        assert deg_2_squared_coeffs.shape[-1] == self.coeffs_count["degree_2_squared"]

        if self.state_dim > 1:
            deg_2_mixed_coeffs = params[..., 1 + 2 * self.state_dim : -1]  # [..., self.state_dim * (self.state_dim - 1) / 2]
            assert deg_2_mixed_coeffs.shape[-1] == self.coeffs_count["degree_2_mixed"]

        else:
            deg_2_mixed_coeffs = None

        scale = params[..., -1].unsqueeze(-1)  # [...,1]

        return const_coeff, deg_1_coeffs, deg_2_squared_coeffs, deg_2_mixed_coeffs, scale

    def evaluate_monomials(self, states) -> tuple[Tensor]:
        """
        Construct and evaluate all monomials up to degree 2 from entries in state.

        Args:
            states (Tensor): State (x_1, ..., x_D). Shape: [..., state_dim]

        Returns:
            evaluated monomials (tuple[Tensor]). Shape: [..., monomial_group_count]
            where monomial_group_count is the number of monomials in each group
        """

        # states.shape == [..., self.state_dim]
        assert states.shape[-1] == self.coeffs_count["degree_1"]

        # constant monomial
        const_mon = torch.ones_like(states[..., 0].unsqueeze(-1))  # [..., 1]

        # degree 1 monomials
        deg_1_mon = states

        # degree 2 monomials with squares, i.e. x_i^2
        deg_2_squared_mon = states**2

        # degree 2 monomials with mixed indices, i.e. x_1 * x_2
        xs = [states[..., i] for i in range(self.state_dim)]
        xs_combs = list(combinations(xs, 2))
        assert len(xs_combs) == self.coeffs_count["degree_2_mixed"]

        if self.state_dim > 1:
            deg_2_mixed_mon = [torch.prod(torch.stack(xs_comb, dim=-1), dim=-1) for xs_comb in xs_combs]
            deg_2_mixed_mon = torch.stack(deg_2_mixed_mon, axis=-1)  # [..., self.state_dim * (self.state_dim - 1) / 2]
            assert deg_2_mixed_mon.shape[-1] == self.coeffs_count["degree_2_mixed"]

        else:
            deg_2_mixed_mon = None

        return const_mon, deg_1_mon, deg_2_squared_mon, deg_2_mixed_mon

    def evaluate_polynomial(self, states: Tensor, coeffs: Tensor) -> Tensor:
        """
        Evaluate polynomials with sampled coefficients, per dimension.

        Args:
            states (Tensor): To evaluate polynomials at. Shape: [..., state_dim]
            coeffs (Tensor): Sampled coefficients for polynomials. Shape: [..., state_dim, total_coeffs_count]

        Returns:
            poly_values (Tensor): Value of polynomials at state. Shape: [..., state_dim]
        """
        assert states.shape[-1] == self.state_dim

        # get monomials and coefficients
        const_mon, deg_1_mon, deg_2_squared_mon, deg_2_mixed_mon = self.evaluate_monomials(states)

        # repeat for all dimensions of polynomial
        const_mon = torch.repeat_interleave(const_mon.unsqueeze(-2), dim=-2, repeats=self.state_dim)
        deg_1_mon = torch.repeat_interleave(deg_1_mon.unsqueeze(-2), dim=-2, repeats=self.state_dim)
        deg_2_squared_mon = torch.repeat_interleave(deg_2_squared_mon.unsqueeze(-2), dim=-2, repeats=self.state_dim)
        if deg_2_mixed_mon is not None:
            deg_2_mixed_mon = torch.repeat_interleave(deg_2_mixed_mon.unsqueeze(-2), dim=-2, repeats=self.state_dim)

        # get coefficient of polynomials for all dimensions
        const_coeff, deg_1_coeffs, deg_2_squared_coeffs, deg_2_mixed_coeffs, scale = self.extract_coefficients(coeffs)

        # evaluate polynomials
        all_terms = [const_coeff * const_mon, deg_1_coeffs * deg_1_mon, deg_2_squared_coeffs * deg_2_squared_mon]

        if self.state_dim > 1:
            all_terms = all_terms + [deg_2_mixed_coeffs * deg_2_mixed_mon]

        all_terms = torch.concatenate(all_terms, dim=-1)
        all_terms = all_terms * scale
        poly_value = torch.sum(all_terms, dim=-1)

        return poly_value

    def print_polynomial(self, coeffs: Tensor, label: str) -> None:
        """
        Print (rounded) string representation of sampled polynomials.

        Args:
            coeffs (Tensor): Sampled coefficients for polynomials. Shape: [num_realizations, state_dim, total_coeffs_count]
            label (str): Drift or diffusion.
        """
        const_coeff, deg_1_coeffs, deg_2_squared_coeffs, deg_2_mixed_coeffs, scale = self.extract_coefficients(coeffs)
        realization_count = const_coeff.shape[0]

        xs = ["x_" + str(i + 1) for i in range(self.state_dim)]

        print("Sampled " + label + ":")
        for realization in range(realization_count):
            print("Realization: ", str(realization))
            for dim in range(self.state_dim):
                # constant term
                const_str = str(round(const_coeff[realization, dim].reshape(-1).item(), self.precision))

                # degree 1 terms
                deg_1_str = [
                    str(round(deg_1_coeffs[realization, dim, i].item(), self.precision)) + " " + xs[i] for i in range(self.state_dim)
                ]
                deg_1_str = " + ".join(deg_1_str)

                # degree 2 squared terms, e.g. x_i^2
                deg_2_squared_str = [
                    str(round(deg_2_squared_coeffs[realization, dim, i].item(), self.precision)) + " " + xs[i] + "^2"
                    for i in range(self.state_dim)
                ]
                deg_2_squared_str = " + ".join(deg_2_squared_str)

                # degree 2 mixed terms, e.g. x_1 * x_2
                if self.state_dim > 1:
                    xs_combs = list(combinations(xs, 2))
                    deg_2_mixed_str = [
                        str(round(deg_2_mixed_coeffs[realization, dim, i].item(), self.precision)) + " " + xs_comb[0] + " " + xs_comb[1]
                        for i, xs_comb in enumerate(xs_combs)
                    ]
                    deg_2_mixed_str = " + ".join(deg_2_mixed_str)

                else:
                    deg_2_mixed_str = None

                # scale
                scale_str = str(round(scale[realization, dim].item(), self.precision))

                # assemble string representation of polynomial
                poly_str = scale_str + " * (" + const_str + " + " + deg_1_str + " + " + deg_2_squared_str
                if self.state_dim > 1:
                    poly_str = poly_str + " + " + deg_2_mixed_str

                poly_str = poly_str + ")"

                print(poly_str)

            print("\n")

    def sample_drift_params(self, num_realizations: int) -> Tensor:
        """
        Sample coefficients of (up to) degree 2 polynomials for each dimension.

        Args:
            num_realizations (int): Number of (polynomial) vector fields to sample.

        Returns:
            params (Tensor): Contains sampled coefficients. Shape: [num_realizations, state_dim, total_coeffs_count]
        """
        if self.drift_params is None or self.drift_params == {}:
            return torch.zeros(num_realizations)

        else:
            drift_params = self.sample_polynomial_coeffs(num_realizations, self.drift_params)

            if self.show_equation is True:
                self.print_polynomial(drift_params, "Drift")

            return drift_params

    def sample_diffusion_params(self, num_realizations: int) -> Tensor:
        """
        Sample coefficients of (up to) degree 2 polynomials for each dimension.

        Args:
            num_realizations (int): Number of (polynomial) vector fields to sample.

        Returns:
            params (Tensor): Contains sampled coefficients. Shape: [num_realizations, state_dim, total_coeffs_count]
        """
        if self.diffusion_params is None or self.diffusion_params == {}:
            return torch.zeros(num_realizations)

        else:
            diffusion_params = self.sample_polynomial_coeffs(num_realizations, self.diffusion_params)

            if self.show_equation is True:
                self.print_polynomial(diffusion_params, "Diffusion")
                print("Diffusion value is Sqrt(" + str(self.enforce_positivity) + "(polynomial)).\n")

            return diffusion_params

    def drift(self, states: Tensor, time: Tensor, params: Tensor) -> Tensor:
        """
        Evaluate (up to) degree 2 polynomials as drift vector field.

        Args:
            states (Tensor): State (x_1, ..., x_D). Shape: [..., state_dim]
            time (Tensor): Time. Shape: [..., 1] (Not used in this system)
            params (Tensor): Contains sampled coefficients. Shape: [..., state_dim, total_coeffs_count]

        Returns:
            drift_poly_values (Tensor): Values of sampled polynomials at state. Shape: [..., state_dim]

        """
        if self.drift_params is None or self.drift_params == {}:
            return torch.zeros_like(states)

        else:
            drift_poly_values = self.evaluate_polynomial(states, params)

            return drift_poly_values

    def diffusion(self, states: Tensor, time: Tensor, params: Tensor) -> Tensor:
        """
        Evaluate (up to) degree 2 polynomials, enforce they are positive and return the squared root as diffusion vector field.

        Args:
            states (Tensor): State (x_1, ..., x_D). Shape: [..., state_dim]
            time (Tensor): Time. Shape: [..., 1] (Not used in this system)
            params (Tensor): Contains sampled coefficients. Shape: [..., state_dim, total_coeffs_count]

        Returns:
            diffusion_values (Tensor): Values of vector field state. Shape: [..., state_dim]

        """
        if self.diffusion_params is None or self.diffusion_params == {}:
            return torch.zeros_like(states)

        else:
            diffusion_poly_values = self.evaluate_polynomial(states, params)

            if self.enforce_positivity == "clip":
                diffusion_values = torch.clip(diffusion_poly_values, min=0)

            elif self.enforce_positivity == "exp":
                diffusion_values = torch.exp(diffusion_poly_values)

            elif self.enforce_positivity == "abs":
                diffusion_values = torch.abs(diffusion_poly_values)

            elif self.enforce_positivity is None:
                raise ValueError("Must pass `enforce_positivity` method to make poly_value values positive.")

            else:
                raise ValueError("`enforce_positivity` method not recognized.")

            return torch.sqrt(diffusion_values)

    def sample_initial_states(self, num_initial_states: int) -> Tensor:
        """
        Sample initial states for the system.

        Args:
            num_initial_states (int): Number of initial states to sample.

        Returns:
            initial_states (Tensor): Initial states. Shape: [num_initial_states, state_dim]

        """
        return self.sample_initial_states_generic(num_initial_states)


class HybridDynamicalSystem(DynamicalSystem):
    # TODO: Remove? Could not be necessary anymore.
    """
    Combine drift and diffusion vector fields from different dynamical systems.
    """

    name_str: str = "HybridDynamicalSystem"

    def __init__(self, config):
        super().__init__(config)

        self.drift_ds: DynamicalSystem = config.get("drift_dynamical_system")
        self.diffusion_ds: DynamicalSystem = config.get("diffusion_dynamical_system")

        self.state_dim = self.drift_ds.state_dim

    def drift(self, states: Tensor, time: Tensor, params: Tensor) -> Tensor:
        return self.drift_ds.drift(states, time, params)

    def diffusion(self, states: Tensor, time: Tensor, params: Tensor) -> Tensor:
        return self.diffusion_ds.diffusion(states, time, params)

    def sample_drift_params(self, num_realizations: int) -> Tensor:
        return self.drift_ds.sample_drift_params(num_realizations)

    def sample_diffusion_params(self, num_realizations: int) -> Tensor:
        return self.diffusion_ds.sample_diffusion_params(num_realizations)

    def sample_initial_states(self, num_initial_states: int) -> Tensor:
        """
        Sample initial states for the system.

        Args:
            num_initial_states (int): Number of initial states to sample.

        Returns:
            initial_states (Tensor): Initial states. Shape: [num_initial_states, state_dim]

        """
        return self.sample_initial_states_generic(num_initial_states)


# ------------------------------------------------------------------------------------------
# MODEL REGISTRY
DYNAMICS_LABELS = {
    "LorenzSystem63": 0,
    "HopfBifurcation": 1,
    "DampedCubicOscillatorSystem": 2,
    "SelkovGlycosis": 3,
    "DuffingOscillator": 4,
    "DampedLinearOscillatorSystem": 5,
    "DoubleWellOneDimension": 6,
    "Degree2Polynomial": 7,
    "HybridDynamicalSystem": 8,
}

REVERSED_DYNAMICS_LABELS = {
    0: "LorenzSystem63",
    1: "HopfBifurcation",
    2: "DampedCubicOscillatorSystem",
    3: "SelkovGlycosis",
    4: "DuffingOscillator",
    5: "DampedLinearOscillatorSystem",
    6: "DoubleWellOneDimension",
    7: "Degree2Polynomial",
    8: "HybridDynamicalSystem",
}

DYNAMICAL_SYSTEM_TO_MODELS = {
    "Lorenz63System": Lorenz63System,
    "HopfBifurcation": HopfBifurcation,
    "DampedCubicOscillatorSystem": DampedCubicOscillatorSystem,
    "DampedLinearOscillatorSystem": DampedLinearOscillatorSystem,
    "SelkovGlycosis": SelkovGlycosis,
    "DuffingOscillator": DuffingOscillator,
    "DoubleWellOneDimension": DoubleWellOneDimension,
    "Degree2Polynomial": Degree2Polynomial,
    "HybridDynamicalSystem": HybridDynamicalSystem,
}
