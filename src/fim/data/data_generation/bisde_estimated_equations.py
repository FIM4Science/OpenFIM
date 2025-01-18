import torch

from fim.data.data_generation.dynamical_systems import DynamicalSystem


def bisde_est_2d_synthetic_drift(x):
    dx_0 = 0.9941 * x[..., 0] - 0.9738 * x[..., 1] - 1.0076 * x[..., 0] * x[..., 1] ** 2 - 1.0294 * x[..., 0] ** 3
    dx_1 = 0.9883 * x[..., 0] + 0.9452 * x[..., 1] - 1.0744 * x[..., 0] ** 2 * x[..., 1] - 0.9387 * x[..., 1] ** 3
    return torch.stack([dx_0, dx_1], dim=-1)


def bisde_est_2d_synthetic_diffusion(x):
    dx_0 = torch.sqrt(1.0050 + 0.9897 * x[..., 1] ** 2)
    dx_1 = torch.sqrt(0.9984 + 1.0306 * x[..., 0] ** 2)
    return torch.stack([dx_0, dx_1], dim=-1)


class BISDEEst2DSynthetic(DynamicalSystem):
    name_str: str = "BISDEEst2DSynthetic"
    state_dim: int = 2

    def __init__(self, config: dict):
        super().__init__(config)

    def drift(self, states, time, params):
        return bisde_est_2d_synthetic_drift(states)

    def diffusion(self, states, time, params):
        return bisde_est_2d_synthetic_diffusion(states)

    def sample_drift_params(self, num_paths):
        return torch.zeros(num_paths, 2)

    def sample_diffusion_params(self, num_paths):
        return torch.zeros(num_paths, 2)

    def sample_initial_states(self, num_paths):
        return torch.ones(num_paths, 2)


def bisde_est_double_well_drift(x):
    return 0.9926 * x - 1.0099 * x**3


def bisde_est_double_well_diffusion(x):
    return torch.sqrt(1.0023 + 0.9783 * x**2)


class BISDEEstDoubleWell(DynamicalSystem):
    name_str: str = "BISDEEstDoubleWell"
    state_dim: int = 1

    def __init__(self, config: dict):
        super().__init__(config)

    def drift(self, states, time, params):
        return bisde_est_double_well_drift(states)

    def diffusion(self, states, time, params):
        return bisde_est_double_well_diffusion(states)

    def sample_drift_params(self, num_paths):
        return torch.zeros(num_paths, 1)

    def sample_diffusion_params(self, num_paths):
        return torch.zeros(num_paths, 1)

    def sample_initial_states(self, num_paths):
        return torch.ones(num_paths, 1)


def sindy_est_double_well_drift(x):
    return 1.0492 * x - 1.0610 * x**3


def sindy_est_double_well_diffusion(x):
    return 1.3858 * torch.ones_like(x)


class SINDyEstDoubleWell(DynamicalSystem):
    name_str: str = "SINDyEstDoubleWell"
    state_dim: int = 1

    def __init__(self, config: dict):
        super().__init__(config)

    def drift(self, states, time, params):
        return sindy_est_double_well_drift(states)

    def diffusion(self, states, time, params):
        return sindy_est_double_well_diffusion(states)

    def sample_drift_params(self, num_paths):
        return torch.zeros(num_paths, 1)

    def sample_diffusion_params(self, num_paths):
        return torch.zeros(num_paths, 1)

    def sample_initial_states(self, num_paths):
        return torch.ones(num_paths, 1)


def bisde_est_facebook_drift(x):
    return 0.0829 * x


def bisde_est_facebook_diffusion(x):
    return 0.4039 * x


class BISDEEstFacebook(DynamicalSystem):
    name_str: str = "BISDEEstFacebook"
    state_dim: int = 1

    def __init__(self, config: dict):
        super().__init__(config)

    def drift(self, states, time, params):
        return bisde_est_facebook_drift(states)

    def diffusion(self, states, time, params):
        return bisde_est_facebook_diffusion(states)

    def sample_drift_params(self, num_paths):
        return torch.zeros(num_paths, 1)

    def sample_diffusion_params(self, num_paths):
        return torch.zeros(num_paths, 1)

    def sample_initial_states(self, num_paths):
        return 229.09 * torch.ones(num_paths, 1)  # initial state of data


def bisde_est_tesla_drift(x):
    return 1.7042 * x


def bisde_est_tesla_diffusion(x):
    return 0.9794 * x


class BISDEEstTesla(DynamicalSystem):
    name_str: str = "BISDEEstTesla"
    state_dim: int = 1

    def __init__(self, config: dict):
        super().__init__(config)

    def drift(self, states, time, params):
        return bisde_est_tesla_drift(states)

    def diffusion(self, states, time, params):
        return bisde_est_tesla_diffusion(states)

    def sample_drift_params(self, num_paths):
        return torch.zeros(num_paths, 1)

    def sample_diffusion_params(self, num_paths):
        return torch.zeros(num_paths, 1)

    def sample_initial_states(self, num_paths):
        return 218.17 * torch.ones(num_paths, 1)  # initial state of data


def bisde_est_wind_drift(x):
    return -6.7949 * x


def bisde_est_wind_diffusion(x):
    return torch.sqrt(2.7433 + 1.7688 * x**2)


class BISDEEstWind(DynamicalSystem):
    name_str: str = "BISDEEstWind"
    state_dim: int = 1

    def __init__(self, config: dict):
        super().__init__(config)

    def drift(self, states, time, params):
        return bisde_est_wind_drift(states)

    def diffusion(self, states, time, params):
        return bisde_est_wind_diffusion(states)

    def sample_drift_params(self, num_paths):
        return torch.zeros(num_paths, 1)

    def sample_diffusion_params(self, num_paths):
        return torch.zeros(num_paths, 1)

    def sample_initial_states(self, num_paths):
        return 9.4 * torch.ones(num_paths, 1)  # initial state of data


def bisde_est_oil_drift(x):
    return -1.0493 * x


def bisde_est_oil_diffusion(x):
    return torch.sqrt(torch.abs(0.3451 * x**2 + 0.3898 * torch.sin(2 * x)))


class BISDEEstOil(DynamicalSystem):
    name_str: str = "BISDEEstOil"
    state_dim: int = 1

    def __init__(self, config: dict):
        super().__init__(config)

    def drift(self, states, time, params):
        return bisde_est_oil_drift(states)

    def diffusion(self, states, time, params):
        return bisde_est_oil_diffusion(states)

    def sample_drift_params(self, num_paths):
        return torch.zeros(num_paths, 1)

    def sample_diffusion_params(self, num_paths):
        return torch.zeros(num_paths, 1)

    def sample_initial_states(self, num_paths):
        return 25.56 * torch.ones(num_paths, 1)  # initial state of data
