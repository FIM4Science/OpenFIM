from typing import Any, Dict

import torch
from torch import Tensor, nn

from ..utils.helper import create_class_instance
from .blocks import AModel, ModelFactory, RNNEncoder, TransformerEncoder
from .blocks.positional_encodings import SineTimeEncoding
from .utils import create_matrix_from_off_diagonal, create_padding_mask, get_off_diagonal_elements


class FIMMJP(AModel):
    """
    FIMMJP: A Neural Recognition Model for Zero-Shot Inference of Markov Jump Processes
    This class implements a neural recognition model for zero-shot inference of Markov jump processes (MJPs) on bounded state spaces from noisy and sparse observations. The methodology is based on the following paper:
    Markov jump processes are continuous-time stochastic processes which describe dynamical systems evolving in discrete state spaces. These processes find wide application in the natural sciences and machine learning, but their inference is known to be far from trivial. In this work we introduce a methodology for zero-shot inference of Markov jump processes (MJPs), on bounded state spaces, from noisy and sparse observations, which consists of two components. First, a broad probability distribution over families of MJPs, as well as over possible observation times and noise mechanisms, with which we simulate a synthetic dataset of hidden MJPs and their noisy observations. Second, a neural recognition model that processes subsets of the simulated observations, and that is trained to output the initial condition and rate matrix of the target MJP in a supervised way. We empirically demonstrate that one and the same (pretrained) recognition model can infer, in a zero-shot fashion, hidden MJPs evolving in state spaces of different dimensionalities. Specifically, we infer MJPs which describe (i) discrete flashing ratchet systems, which are a type of Brownian motors, and the conformational dynamics in (ii) molecular simulations, (iii) experimental ion channel data and (iv) simple protein folding models. What is more, we show that our model performs on par with state-of-the-art models which are trained on the target datasets.

    It is model from the paper:"Foundation Inference Models for Markov Jump Processes" --- https://arxiv.org/abs/2406.06419.
    Attributes:
        n_states (int): Number of states in the Markov jump process.
        use_adjacency_matrix (bool): Whether to use an adjacency matrix.
        ts_encoder (dict | TransformerEncoder): Time series encoder.
        pos_encodings (dict | SineTimeEncoding): Positional encodings.
        path_attention (dict | nn.Module): Path attention mechanism.
        intensity_matrix_decoder (dict | nn.Module): Decoder for the intensity matrix.
        initial_distribution_decoder (dict | nn.Module): Decoder for the initial distribution.
        gaussian_nll (nn.GaussianNLLLoss): Gaussian negative log-likelihood loss.
        init_cross_entropy (nn.CrossEntropyLoss): Cross-entropy loss for initial distribution.

    Methods:
        forward(x: dict[str, Tensor], schedulers: dict = None, step: int = None) -> dict:
            Forward pass of the model.
        __decode(h: Tensor) -> tuple[Tensor, Tensor]:
            Decode the hidden representation to obtain the intensity matrix and initial condition.
        __encode(x: Tensor, obs_grid_normalized: Tensor, obs_values_one_hot: Tensor) -> Tensor:
            Encode the input observations to obtain the hidden representation.
        __denormalize_offdiag_mean_logvar(norm_constants: Tensor, pred_offdiag_im_mean_logvar: Tensor) -> tuple[Tensor, Tensor]:
            Denormalize the predicted off-diagonal mean and log-variance.
        __normalize_obs_grid(obs_grid: Tensor) -> tuple[Tensor, Tensor]:
            Normalize the observation grid.
        loss(pred_im: Tensor, pred_logvar_im: Tensor, pred_init_cond: Tensor, target_im: Tensor, target_init_cond: Tensor, adjaceny_matrix: Tensor, normalization_constants: Tensor, schedulers: dict = None, step: int = None) -> dict:
            Compute the loss for the model.
        new_stats() -> dict:
            Initialize new statistics.
        metric(y: Any, y_target: Any) -> Dict:
            Compute the metric for the model.
    """

    def __init__(
        self,
        n_states: int,
        use_adjacency_matrix: bool,
        ts_encoder: dict | TransformerEncoder,
        pos_encodings: dict | SineTimeEncoding,
        path_attention: dict | nn.Module,
        intensity_matrix_decoder: dict | nn.Module,
        initial_distribution_decoder: dict | nn.Module,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_states = n_states
        self.use_adjacency_matrix = use_adjacency_matrix
        self.ts_encoder = ts_encoder
        total_offdiagonal_transitions = n_states**2 - n_states

        self.__create_modules(
            n_states,
            use_adjacency_matrix,
            ts_encoder,
            pos_encodings,
            path_attention,
            intensity_matrix_decoder,
            initial_distribution_decoder,
            total_offdiagonal_transitions,
        )

        self.gaussian_nll = nn.GaussianNLLLoss(full=True, reduction="none")
        self.init_cross_entropy = nn.CrossEntropyLoss(reduction="none")

    def __create_modules(
        self,
        n_states,
        use_adjacency_matrix,
        ts_encoder,
        pos_encodings,
        path_attention,
        intensity_matrix_decoder,
        initial_distribution_decoder,
        total_offdiagonal_transitions,
    ):
        self.pos_encodings = pos_encodings
        if isinstance(pos_encodings, dict):
            if isinstance(ts_encoder, dict) and ts_encoder["name"] == "fim.models.blocks.base.TransformerEncoder":
                pos_encodings["out_features"] -= self.n_states
            self.pos_encodings = create_class_instance(pos_encodings.pop("name"), pos_encodings)

        self.ts_encoder = ts_encoder
        if isinstance(ts_encoder, dict):
            ts_encoder["in_features"] = n_states + self.pos_encodings.out_features
            self.ts_encoder = create_class_instance(ts_encoder.pop("name"), ts_encoder)

        self.path_attention = path_attention
        if isinstance(path_attention, dict):
            self.path_attention = create_class_instance(path_attention.pop("name"), path_attention)

        if isinstance(intensity_matrix_decoder, dict):
            in_features = intensity_matrix_decoder.get(
                "in_features", self.ts_encoder.out_features + ((total_offdiagonal_transitions + 1) if use_adjacency_matrix else 1)
            )
            intensity_matrix_decoder["in_features"] = in_features
            intensity_matrix_decoder["out_features"] = 2 * total_offdiagonal_transitions
            self.intensity_matrix_decoder = create_class_instance(intensity_matrix_decoder.pop("name"), intensity_matrix_decoder)

        if isinstance(initial_distribution_decoder, dict):
            in_features = initial_distribution_decoder.get(
                "in_features", self.ts_encoder.out_features + ((total_offdiagonal_transitions + 1) if use_adjacency_matrix else 1)
            )
            initial_distribution_decoder["in_features"] = in_features
            initial_distribution_decoder["out_features"] = n_states
            self.initial_distribution_decoder = create_class_instance(
                initial_distribution_decoder.pop("name"), initial_distribution_decoder
            )

    def forward(self, x: dict[str, Tensor], schedulers: dict = None, step: int = None) -> dict:
        """
        Forward pass for the model.

        Args:
            x (dict[str, Tensor]): A dictionary containing the input tensors:
                - "observation_grid": Tensor representing the observation grid.
                - "observation_values": Tensor representing the observation values.
                - Optional keys for loss calculation:
                    - "intensity_matrices": Tensor representing the intensity matrices.
                    - "initial_distributions": Tensor representing the initial distributions.
                    - "adjacency_matrices": Tensor representing the adjacency matrices.
            schedulers (dict, optional): A dictionary of schedulers for the training process. Default is None.
            step (int, optional): The current step in the training process. Default is None.
        Returns:
            dict: A dictionary containing the following keys:
                - "im": Tensor representing the intensity matrix.
                - "log_var_im": Tensor representing the log variance of the intensity matrix.
                - "init_cond": Tensor representing the initial conditions.
                - "losses" (optional): Tensor representing the calculated losses, if the required keys are present in `x`.
        """

        obs_grid = x["observation_grid"]
        if "time_normalization_factors" not in x:
            norm_constants, obs_grid = self.__normalize_obs_grid(obs_grid)
        else:
            norm_constants = x["time_normalization_factors"]

        obs_values_one_hot = torch.nn.functional.one_hot(x["observation_values"].long().squeeze(-1), num_classes=self.n_states)

        h = self.__encode(x, obs_grid, obs_values_one_hot)

        pred_offdiag_im_mean_logvar, init_cond = self.__decode(h)

        pred_offdiag_im_mean, pred_offdiag_im_logvar = self.__denormalize_offdiag_mean_logvar(norm_constants, pred_offdiag_im_mean_logvar)

        out = {
            "im": create_matrix_from_off_diagonal(pred_offdiag_im_mean, self.n_states),
            "log_var_im": create_matrix_from_off_diagonal(pred_offdiag_im_logvar, self.n_states),
            "init_cond": init_cond,
        }
        if "intensity_matrices" in x and "initial_distributions" in x:
            out["losses"] = self.loss(
                pred_offdiag_im_mean, pred_offdiag_im_logvar, init_cond, x, norm_constants.view(-1, 1), schedulers, step
            )

        return out

    def __decode(self, h: Tensor) -> tuple[Tensor, Tensor]:
        pred_offdiag_mean_logvar = self.intensity_matrix_decoder(h)
        init_cond = self.initial_distribution_decoder(h)
        return pred_offdiag_mean_logvar, init_cond

    def __encode(self, x: Tensor, obs_grid_normalized: Tensor, obs_values_one_hot: Tensor) -> Tensor:
        B, P, L = obs_grid_normalized.shape[:3]
        pos_enc = self.pos_encodings(obs_grid_normalized)
        path = torch.cat([pos_enc, obs_values_one_hot], dim=-1)
        if isinstance(self.ts_encoder, TransformerEncoder):
            padding_mask = create_padding_mask(x["mask_seq_lengths"].view(B * P), L)
            padding_mask[:, 0] = True
            h = self.ts_encoder(path.view(B * P, L, -1), padding_mask)[:, 1, :].view(B, P, -1)
            if isinstance(self.path_attention, nn.MultiheadAttention):
                h = self.path_attention(h, h, h)[0][:, -1]
            else:
                h = self.path_attention(h, h, h)
        elif isinstance(self.ts_encoder, RNNEncoder):
            h = self.ts_encoder(path.view(B * P, L, -1), x["mask_seq_lengths"].view(B * P))
            last_observation = x["mask_seq_lengths"].view(B * P) - 1
            h = h[torch.arange(B * P), last_observation].view(B, P, -1)
            h = self.path_attention(h, h, h)

        h = torch.cat([h, torch.ones(B, 1).to(h.device) / 100.0 * P], dim=-1)
        if self.use_adjacency_matrix:
            h = torch.cat([h, get_off_diagonal_elements(x["adjacency_matrix"])], dim=-1)
        return h

    def __denormalize_offdiag_mean_logvar(self, norm_constants: Tensor, pred_offdiag_im_mean_logvar: Tensor) -> tuple[Tensor, Tensor]:
        pred_offdiag_im_logmean, pred_offdiag_im_logvar = pred_offdiag_im_mean_logvar.chunk(2, dim=-1)
        pred_offdiag_im_mean = torch.exp(pred_offdiag_im_logmean) / norm_constants.view(-1, 1)
        pred_offdiag_im_logvar = pred_offdiag_im_logvar - torch.log(norm_constants.view(-1, 1))
        return pred_offdiag_im_mean, pred_offdiag_im_logvar

    def __normalize_obs_grid(self, obs_grid: Tensor) -> tuple[Tensor, Tensor]:
        norm_constants = obs_grid.amax(dim=[-3, -2, -1])
        obs_grid_normalized = obs_grid / norm_constants.view(-1, 1, 1, 1)
        return norm_constants, obs_grid_normalized

    def loss(
        self,
        pred_im: Tensor,
        pred_logvar_im: Tensor,
        pred_init_cond: Tensor,
        target: dict,
        normalization_constants: Tensor,
        schedulers: dict = None,
        step: int = None,
    ) -> dict:
        target_im = target["intensity_matrices"]
        target_init_cond = target["initial_distributions"]
        adjaceny_matrix = target["adjacency_matrices"]
        target_mean = get_off_diagonal_elements(target_im)
        adjaceny_matrix = get_off_diagonal_elements(adjaceny_matrix)
        target_init_cond = torch.argmax(target_init_cond, dim=-1).long()
        pred_im_var = torch.exp(pred_logvar_im)
        loss_gauss = adjaceny_matrix * self.gaussian_nll(pred_im, target_mean, pred_im_var)
        loss_initial = self.init_cross_entropy(pred_init_cond, target_init_cond)
        zero_entries = 1.0 - adjaceny_matrix
        loss_missing_link = normalization_constants * zero_entries * (torch.pow(pred_im, 2) + pred_im_var)
        rmse_loss = torch.sqrt(torch.mean((target_mean - pred_im) ** 2))

        gaus_cons = schedulers.get("gauss_nll")(step) if schedulers else torch.tensor(1.0)
        init_cons = schedulers.get("init_cross_entropy")(step) if schedulers else torch.tensor(1.0)
        missing_link_cons = schedulers.get("missing_link")(step) if schedulers else torch.tensor(1.0)
        gaus_cons = gaus_cons.to(self.device)
        init_cons = init_cons.to(self.device)
        missing_link_cons = missing_link_cons.to(self.device)

        loss = gaus_cons * loss_gauss + init_cons * loss_initial.view(-1, 1) + missing_link_cons * loss_missing_link

        return {
            "loss": loss.sum(),
            "loss_gauss": loss_gauss.sum() / adjaceny_matrix.sum(),
            "loss_initial": loss_initial.mean(),
            "loss_missing_link": loss_missing_link.sum() / zero_entries.sum(),
            "rmse_loss": rmse_loss,
            "beta_gauss_nll": gaus_cons,
            "beta_init_cross_entropy": init_cons,
            "beta_missing_link": missing_link_cons,
        }

    def metric(self, y: Any, y_target: Any) -> Dict:
        return super().metric(y, y_target)


ModelFactory.register("FIMMJP", FIMMJP)
