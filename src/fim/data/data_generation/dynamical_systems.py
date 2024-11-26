import torch


from abc import ABC, abstractmethod
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
}

REVERSED_DYNAMICS_LABELS = {
    0: "LorenzSystem63",
    1: "HopfBifurcation",
    2: "DampedCubicOscillatorSystem",
    3: "SelkovGlycosis",
    4: "DuffingOscillator",
    5: "DampedLinearOscillatorSystem",
    6: "DoubleWellOneDimension",
}

DYNAMICAL_SYSTEM_TO_MODELS = {
    "Lorenz63System": Lorenz63System,
    "HopfBifurcation": HopfBifurcation,
    "DampedCubicOscillatorSystem": DampedCubicOscillatorSystem,
    "DampedLinearOscillatorSystem": DampedLinearOscillatorSystem,
    "SelkovGlycosis": SelkovGlycosis,
    "DuffingOscillator": DuffingOscillator,
    "DoubleWellOneDimension": DoubleWellOneDimension,
}
