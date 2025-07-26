from copy import deepcopy
from typing import Any, Dict

import torch
import torchsde
from torch import Tensor, nn
from transformers import AutoConfig, AutoModel, PretrainedConfig

from fim.models.blocks import AModel, ModelFactory


class Encoder(nn.Module):
    """
    Taken from: https://github.com/google-research/torchsde/blob/master/examples/latent_sde_lorenz.py
    """

    def __init__(self, input_size, hidden_size, output_size):
        super(Encoder, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size)
        self.lin = nn.Linear(hidden_size, output_size)

    def forward(self, inp):
        out, _ = self.gru(inp)
        out = self.lin(out)
        return out


class LatentSDEConfig(PretrainedConfig):
    """
    LatentSDEConfig is a configuration class for the LatentSDE model.

    Attributes:
        name (str): Name of the configuration. Default is "LatentSDE".
        data_size (int): Size of input data. Default is 3.
        hidden_size (int): Hidden size of GRU in encoder and vector field networks. Default is 32.
        context_size (int): Output size of encoder. Default is 32.
        latent_size (int): Size of SDE. Default is 32.
        activation (str): Describes activation function for MLPs. Default is sigmoid.
        noise_std (float): Fixed standard deviation for decoder. Default is 0.01
        solver_adjoint (bool): Flag to use adjoint method for solving SDE during training. Default is False.
        solver_method (str): Method to solve SDE during training, passed to torchsde.sdeint. Default is euler.
        solver_dt (float): Generation step size during training. Default is 0.02.
        learn_projection (bool): Flag to learn a output projection MLP or use the identity function. Default is True.
    """

    model_type = "latentsde"

    def __init__(
        self,
        name: str = "LatentSDE",
        model_type: str = "latentsde",
        data_size: int = 3,
        hidden_size: int = 32,
        context_size: int = 32,
        latent_size: int = 32,
        activation: str = "sigmoid",
        noise_std: float = 0.01,
        mse_objective: bool = False,
        solver_adjoint: bool = False,
        solver_method: str = "euler",
        solver_dt: float = 0.02,
        learn_projection: bool = True,
        **kwargs,
    ):
        self.name = name
        self.model_type = model_type
        self.data_size = data_size
        self.hidden_size = hidden_size
        self.context_size = context_size
        self.latent_size = latent_size
        self.activation = activation
        self.noise_std = noise_std
        self.mse_objective = mse_objective
        self.solver_adjoint = solver_adjoint
        self.solver_method = solver_method
        self.solver_dt = solver_dt
        self.learn_projection = learn_projection
        super().__init__(**kwargs)


class LatentSDE(AModel):
    """
    LatentSDE, adapted from: https://github.com/google-research/torchsde/blob/master/examples/latent_sde_lorenz.py
    """

    config_class = LatentSDEConfig
    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(
        self,
        config: LatentSDEConfig,
        device_map: torch.device = None,
        **kwargs,
    ):
        AModel.__init__(self, config, **kwargs)

        if isinstance(config, dict):
            self.config = LatentSDEConfig(**config)
        else:
            self.config = config

        self._create_modules()

        if device_map is not None:
            self.to(device_map)

    def _create_modules(self):
        config = deepcopy(self.config)  # model loading won't work without it

        # Encoder.
        self.encoder = Encoder(input_size=config.data_size, hidden_size=config.hidden_size, output_size=config.context_size)
        self.qz0_net = nn.Linear(config.context_size, config.latent_size + config.latent_size)

        # Decoder.
        if config.activation == "sigmoid":
            activation_fn = nn.Sigmoid
        elif config.activation == "softplus":
            activation_fn = nn.Softplus
        else:
            raise ValueError("activation function not recognized")

        self.f_net = nn.Sequential(
            nn.Linear(config.latent_size + config.context_size, config.hidden_size),
            activation_fn(),
            nn.Linear(config.hidden_size, config.hidden_size),
            activation_fn(),
            nn.Linear(config.hidden_size, config.latent_size),
        )
        self.h_net = nn.Sequential(
            nn.Linear(config.latent_size, config.hidden_size),
            activation_fn(),
            nn.Linear(config.hidden_size, config.hidden_size),
            activation_fn(),
            nn.Linear(config.hidden_size, config.latent_size),
        )
        # This needs to be an element-wise function for the SDE to satisfy diagonal noise.
        self.g_nets = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(1, config.hidden_size),
                    activation_fn(),
                    nn.Linear(config.hidden_size, 1),
                    nn.Sigmoid(),
                )
                for _ in range(config.latent_size)
            ]
        )

        if config.learn_projection is True:
            self.projector = nn.Linear(config.latent_size, config.data_size)

        else:
            assert config.latent_size == config.data_size, "Without learned projection, latent and data space must be the same."
            self.projector = lambda x: x

        self.pz0_mean = nn.Parameter(torch.zeros(1, config.latent_size))
        self.pz0_logstd = nn.Parameter(torch.zeros(1, config.latent_size))

        self._ctx = None

    def f(self, t, y):
        ts, ctx = self._ctx
        i = min(torch.searchsorted(ts, t, right=True), len(ts) - 1)
        return self.f_net(torch.cat((y, ctx[i]), dim=1))

    def h(self, t, y):
        return self.h_net(y)

    def g(self, t, y):  # Diagonal diffusion.
        y = torch.split(y, split_size_or_sections=1, dim=1)
        out = [g_net_i(y_i) for (g_net_i, y_i) in zip(self.g_nets, y)]
        return torch.cat(out, dim=1)

    @torch.profiler.record_function("preprocess_input_times")
    def preprocess_input_times(self, times: Tensor) -> Tensor:
        """
        Require that time grid for all batch elements are the same, as times are not passed to GRU encoder.
        Reduce Tensor to one dimension.

        Args:
            times (Tensor): Observation time points. Shape: [B, 1, T, 1] or [B, T, 1] or [T, 1]

        Returns:
            times (Tensor): Reduced times. Shape: [T]
        """
        assert times.ndim <= 4, f"Got {times.ndim}."

        if times.ndim == 4:  # [B, 1, T, 1]
            assert times.shape[1] == 1, f"Got {times.shape}."
            times = times.squeeze(1)

        if times.ndim == 3:  # [B, T, 1]
            ref = torch.broadcast_to(times[0].unsqueeze(0), times.shape)
            assert torch.allclose(times, ref)
            times = times[0]

        if times.ndim == 2:  # [T, 1]
            assert times.shape[1] == 1, f"Got {times.shape}."
            times = times.squeeze(1)

        return times

    @torch.profiler.record_function("preprocess_input_values")
    def preprocess_input_values(self, values: Tensor) -> Tensor:
        """
        Remove optional path dimension from values.
        Reorder dimensions, as torchsde requires batch in second dimension.

        Args:
            values (Tensor): Observation values. Shape: [B, 1, T, D] or [B, T, D] or [T, D]

        Returns:
            values (Tensor): Reordered values. Shape: [T, B, D]
        """
        assert (values.ndim <= 4) and (values.ndim > 1), f"Got {values.ndim}."

        if values.ndim == 4:  # [B, 1, T, D]
            assert values.shape[1] == 1, f"Got {values.shape}. Can't process multiple paths per batch element."
            values = values.squeeze(1)

        if values.ndim == 3:  # [B, T, D]
            values = torch.transpose(values, 0, 1)  # [T, B, D]

        if values.ndim == 2:  # [T, D]
            values = values[:, None, :]  # [T, 1, D]

        return values

    @torch.profiler.record_function("latentsde_forward")
    def forward(self, data: dict, schedulers: dict | None = None, step: int = 0) -> dict:
        """
        Sample posterior paths and compute train loss.

        Args:
            data (dict): Contains keys `obs_times` (Shape: [B, 1, T, 1] or [T]) and `obs_values` (Shape: [B, 1, T, 3])
            schedulers (dict): Contains key `kl_scale`.
            step (int): Number of prior optimization steps.

        Returns:
            loss_dict (dict): In key `losses`: dict with `loss` (train loss) and aux. losses for monitoring.
        """
        ctx, obs_times, _ = self.encode_inputs(data["obs_times"], data["obs_values"])
        initial_states, qz0_mean, qz0_logstd = self.sample_posterior_initial_condition(ctx[0])
        _, projected_paths, log_ratio = self.sample_from_posterior_equation(
            initial_states,
            ctx,
            obs_times,
            logqp=True,
            solver_dt=self.config.solver_dt,
            solver_adjoint=self.config.solver_adjoint,
            solver_method=self.config.solver_method,
        )

        if schedulers is not None and "kl_scale" in schedulers.keys():
            kl_scale = schedulers.get("kl_scale")(step)

        else:
            kl_scale = 1

        return {"losses": self.loss(data["obs_values"], projected_paths, qz0_mean, qz0_logstd, log_ratio, kl_scale)}

    @torch.profiler.record_function("latentsde_train_loss")
    def loss(self, obs_values: Tensor, pred_values: Tensor, qz0_mean: Tensor, qz0_logstd: Tensor, log_ratio: Tensor, kl_scale: float):
        """
        Approximates the ELBO, where the KL term is weighted by some scale.

        Args:
            obs_values (Tensor): Observations. Shape: [B, 1, T, D] (or similar)
            pred_values (Tensor): Predicted reconstruction of observations. Shape: [B, T, D]
            qz0_mean, qz0_logstd (Tensor): Define posterior initial condition in latent space. Shape: [B, latent_size]
            log_ratio (Tensor): Defines path term  in KL. Shape: [B, T].
            kl_scale (float): Weighting of KL term in ELBO.

        Returns:
            losses (dict): ELBO and individual terms for monitoring. All of Shape: [1]
        """
        obs_values = self.preprocess_input_values(obs_values)  # [T, B, D]
        obs_values = torch.transpose(obs_values, 0, 1)  # [B, T, D]

        pred_dist = torch.distributions.Normal(loc=pred_values, scale=self.config.noise_std)
        qz0 = torch.distributions.Normal(loc=qz0_mean, scale=qz0_logstd.exp())
        pz0 = torch.distributions.Normal(loc=self.pz0_mean, scale=self.pz0_logstd.exp())

        assert obs_values.shape == pred_values.shape
        mse = ((obs_values - pred_values) ** 2).mean()
        nll = -pred_dist.log_prob(obs_values).sum(dim=(1, 2)).mean(dim=0)

        rec_loss = mse if self.config.mse_objective is True else nll

        kl_init_cond = torch.distributions.kl_divergence(qz0, pz0).sum(dim=1).mean(dim=0)
        kl_path = log_ratio.sum(dim=1).mean(dim=0)
        kl = kl_init_cond + kl_path

        loss = rec_loss + kl_scale * kl

        return {"loss": loss, "kl_init_cond": kl_init_cond, "kl_path": kl_path, "nll": nll, "mse": mse, "kl_scale": kl_scale}

    def encode_inputs(self, obs_times: Tensor, obs_values: Tensor):
        """
        Preprocess inputs, to unify shapes, and apply encoder module.

        Args:
            obs_values (Tensor): Observations. Shape: [B, 1, T, D] (or similar)
            sol_times (Tensor): Time grid to extract paths on. Shape: [T] (or similar)

        Returns:
            ctx: Context for posterior sampling. Shape: [T, B, context_size]
            pre_proc_obs_times/values (Tensor): Preprocessed inputs. Shapes: [T], [T, B, D]
        """
        obs_times = self.preprocess_input_times(obs_times)  # [T]
        obs_values = self.preprocess_input_values(obs_values)  # [T, B, D]
        T, B, D = obs_values.shape

        assert obs_times.shape[0] == T, f"Got {obs_times.shape[0]} and {T}."
        assert D == self.config.data_size, f"Got {D} and {self.config.data_size}"

        ctx = self.encoder(torch.flip(obs_values, dims=(0,)))
        ctx = torch.flip(ctx, dims=(0,))
        assert ctx.shape == (T, B, self.config.context_size), f"Got {ctx.shape} and {(T, B, self.config.context_size)}"

        self._ctx = (obs_times, ctx)  # A tuple of tensors of sizes (T,), (T, B, context_size).

        return ctx, obs_times, obs_values

    def sample_posterior_initial_condition(self, ctx: Tensor):
        """
        One sample from posterior initial condition per batch element.

        Args:
            ctx: Context for posterior sampling at first observation. Shape: [B, context_size]

        Returns:
            initial_states (Tensor): Samples from posterior initial condition. Shape: [B, latent_size]
            qz0_mean (Tensor): Posterior mean. Shape: [B, latent_size]
            qz0_logstd (Tensor): Log of posterior standard deviation. Shape: [B, latent_size]
        """
        qz0_mean, qz0_logstd = self.qz0_net(ctx).chunk(chunks=2, dim=1)
        initial_states = qz0_mean + qz0_logstd.exp() * torch.randn_like(qz0_mean)

        B = ctx.shape[0]
        assert initial_states.shape == (B, self.config.latent_size), f"Got {initial_states.shape} and {(B, self.config.latent_size)}."

        return initial_states, qz0_mean, qz0_logstd

    def sample_prior_initial_condition(self, num_initial_states: int):
        """
        num_initial_states samples from (learned) prior initial condition.

        Args:
            num_initial_states (int): Number of samples from initial condition.

        Returns:
            initial_states (Tensor): Samples from prior initial condition. Shape: [B, latent_size]
        """
        eps = torch.randn(size=(num_initial_states, self.config.latent_size), device=self.pz0_mean.device)
        initial_states = self.pz0_mean + self.pz0_logstd.exp() * eps

        return initial_states  # [num_initial_states, latent_size]

    def sample_from_posterior_equation(
        self,
        initial_states,
        ctx,
        sol_times,
        logqp: bool = False,
        solver_dt: float = 1e-3,
        solver_adjoint: bool = False,
        solver_method: str | None = None,
        bm=None,
    ):
        """
        Sample paths starting at given initial states in latent space with the posterior equation inferred from context embedding.

        Args:
            initial_states (Tensor): Initial states in latent space. Shape: [B, latent_size]
            ctx: Context for posterior sampling. Shape: [T, B, context_size]
            sol_times (Tensor): Time grid to extract paths on. Shape: [T]
            logqp (bool): Flag passed to solver. Required for KL. Default is False.
            solver config

        Returns:
            latent_paths (Tensor): Sampled paths in latent space. Shape: [B, T, latent_size]
            projected_paths (Tensor): Sampled paths in data space. Shape: [B, T, data_size]
            log_ratio (Tensor): Defines path term in KL. Shape: [B, T].
        """
        if solver_adjoint:
            adjoint_params = (ctx,) + tuple(self.f_net.parameters()) + tuple(self.g_nets.parameters()) + tuple(self.h_net.parameters())
            solver_out = torchsde.sdeint_adjoint(
                self,
                initial_states,
                sol_times,
                adjoint_params=adjoint_params,
                dt=solver_dt,
                logqp=logqp,
                method=solver_method,
                bm=bm,
            )
        else:
            solver_out = torchsde.sdeint(self, initial_states, sol_times, dt=solver_dt, logqp=logqp, method=solver_method, bm=bm)

        if logqp is True:
            latent_paths, log_ratio = solver_out
        else:
            latent_paths = solver_out
            log_ratio = None

        projected_paths = self.projector(latent_paths)

        T = sol_times.shape[0]
        B = initial_states.shape[0]
        assert latent_paths.shape == (T, B, self.config.latent_size), f"Got {latent_paths.shape} and {(T, B, self.config.context_size)}."
        assert projected_paths.shape == (T, B, self.config.data_size), f"Got {projected_paths.shape} and {(T, B, self.config.data_size)}."
        assert log_ratio.shape == (T - 1, B), f"Got {log_ratio.shape} and {(T - 1, B)}."

        # batch first convention
        latent_paths, projected_paths, log_ratio = torch.utils._pytree.tree_map(
            lambda x: torch.transpose(x, 0, 1), (latent_paths, projected_paths, log_ratio)
        )

        return latent_paths, projected_paths, log_ratio

    def sample_from_prior_equation(
        self,
        initial_states,
        sol_times,
        solver_dt: float = 1e-3,
        solver_method: str | None = None,
        bm=None,
    ):
        """
        Sample paths starting at given initial states in latent space with the learned prior equation.

        Args:
            initial_states (Tensor): Initial states in latent space. Shape: [B, latent_size]
            sol_times (Tensor): Time grid to extract paths on. Shape: [T]
            solver config

        Returns:
            latent_paths (Tensor): Sampled paths in latent space. Shape: [B, T, latent_size]
            projected_paths (Tensor): Sampled paths in data space. Shape: [B, T, data_size]
        """
        latent_paths = torchsde.sdeint(self, initial_states, sol_times, names={"drift": "h"}, method=solver_method, dt=solver_dt, bm=bm)
        projected_paths = self.projector(latent_paths)

        T = sol_times.shape[0]
        B = initial_states.shape[0]
        assert latent_paths.shape == (T, B, self.config.latent_size), f"Got {latent_paths.shape} and {(T, B, self.config.latent_size)}."
        assert projected_paths.shape == (T, B, self.config.data_size), f"Got {projected_paths.shape} and {(T, B, self.config.data_size)}."

        # batch first convention
        latent_paths, projected_paths = torch.utils._pytree.tree_map(lambda x: torch.transpose(x, 0, 1), (latent_paths, projected_paths))

        return latent_paths, projected_paths

    def metric(self, y: Any, y_target: Any) -> Dict:
        return super().metric(y, y_target)


ModelFactory.register(LatentSDEConfig.model_type, LatentSDE)
AutoConfig.register(LatentSDEConfig.model_type, LatentSDEConfig)
AutoModel.register(LatentSDEConfig, LatentSDE)
