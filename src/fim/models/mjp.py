from typing import Any, Dict

import torch
from torch import nn

from ..utils.helper import create_class_instance
from .blocks import AModel, ModelFactory, SineTimeEncoding, TransformerEncoder
from .utils import create_matrix_from_off_diagonal, get_off_diagonal_elements


class FIMMJP(AModel):
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
        r"""_summary_

        Args:
            n_states (int): _description_
            use_adjacency_matrix (bool): _description_
            ts_encoder (dict | TransformerEncoder): It corresponds to \psi_1 in the paper.
            pos_encodings (dict | SineTimeEncoding): _description_
            path_attention (dict | nn.Module): _description_
            intensity_matrix_decoder (dict | nn.Module): _description_
            initial_distribution_decoder (dict | nn.Module): _description_
        """
        super().__init__(**kwargs)
        self.n_states = n_states
        self.use_adjacency_matrix = use_adjacency_matrix
        self.ts_encoder = ts_encoder
        total_offdiagonal_transitions = n_states**2 - n_states
        if isinstance(ts_encoder, dict):
            self.ts_encoder = create_class_instance(ts_encoder.pop("name"), ts_encoder)
        self.pos_encodings = pos_encodings
        if isinstance(pos_encodings, dict):
            pos_encodings["model_dim"] -= self.n_states
            self.pos_encodings = create_class_instance(pos_encodings.pop("name"), pos_encodings)
        self.path_attention = path_attention
        if isinstance(path_attention, dict):
            self.path_attention = create_class_instance(path_attention.pop("name"), path_attention)
        if isinstance(intensity_matrix_decoder, dict):
            intensity_matrix_decoder["in_features"] = self.ts_encoder.model_dim + (
                (total_offdiagonal_transitions + 1) if use_adjacency_matrix else 1
            )
            intensity_matrix_decoder["out_features"] = 2 * total_offdiagonal_transitions
            self.intensity_matrix_decoder = create_class_instance(intensity_matrix_decoder.pop("name"), intensity_matrix_decoder)
        if isinstance(initial_distribution_decoder, dict):
            initial_distribution_decoder["in_features"] = self.ts_encoder.model_dim + (
                (total_offdiagonal_transitions + 1) if use_adjacency_matrix else 1
            )
            initial_distribution_decoder["out_features"] = n_states
            self.initial_distribution_decoder = create_class_instance(
                initial_distribution_decoder.pop("name"), initial_distribution_decoder
            )
        self.gaussian_nll = nn.GaussianNLLLoss(full=True, reduction="none")
        self.init_cross_entropy = nn.CrossEntropyLoss(reduction="none")

    def forward(self, x: dict[str, torch.Tensor], schedulers: dict = None, step: int = None) -> dict:
        obs_grid = x["observation_grid"]
        B, P, L = obs_grid.shape[:3]
        norm_constants = obs_grid.amax(dim=[-3, -2, -1])
        obs_grid_normalized = obs_grid / norm_constants.view(-1, 1, 1, 1)
        obs_values_one_hot = torch.nn.functional.one_hot(x["observation_values"].long().squeeze(-1), num_classes=self.n_states).float()
        pos_enc = self.pos_encodings(obs_grid_normalized)
        path = torch.cat([obs_values_one_hot, pos_enc], dim=-1)  # [t, delta_t]
        h = self.ts_encoder(path.view(B * P, L, -1))[:, -1, :].view(B, P, -1)
        h = self.path_attention(h, h, h)[0][:, -1, :]

        h = torch.cat([h, torch.ones(B, 1).to(h.device) / 100.0 * P], dim=-1)
        if self.use_adjacency_matrix:
            h = torch.cat([h, get_off_diagonal_elements(x["adjacency_matrix"])], dim=-1)
        predicted_offdiagonal_im = self.intensity_matrix_decoder(h)
        init_cond = self.initial_distribution_decoder(h)
        predicted_offdiagonal_im_logmean, predicted_offdiagonal_im_logvar = predicted_offdiagonal_im.chunk(2, dim=-1)
        predicted_offdiagonal_im = torch.exp(predicted_offdiagonal_im_logmean) / norm_constants.view(-1, 1)
        predicted_offdiagonal_im_logvar = predicted_offdiagonal_im_logvar - torch.log(norm_constants.view(-1, 1))

        out = {
            "im": create_matrix_from_off_diagonal(predicted_offdiagonal_im, self.n_states),
            "log_var_im": create_matrix_from_off_diagonal(predicted_offdiagonal_im_logvar, self.n_states),
            "init_cond": init_cond,
        }
        if "intensity_matrices" in x and "initial_distributions" in x:
            out["losses"] = self.loss(
                predicted_offdiagonal_im,
                predicted_offdiagonal_im_logvar,
                init_cond,
                x["intensity_matrices"],
                x["initial_distributions"],
                x["adjacency_matrices"],
                norm_constants.view(-1, 1),
                schedulers,
                step,
            )

        return out

    def loss(
        self,
        pred_im: torch.Tensor,
        pred_logvar_im: torch.Tensor,
        pred_init_cond: torch.Tensor,
        target_im: torch.Tensor,
        target_init_cond: torch.Tensor,
        adjaceny_matrix: torch.Tensor,
        normalization_constants: torch.Tensor,
        schedulers: dict = None,
        step: int = None,
    ) -> dict:
        target_mean = get_off_diagonal_elements(target_im)
        adjaceny_matrix = get_off_diagonal_elements(adjaceny_matrix)
        target_init_cond = torch.argmax(target_init_cond, dim=-1).long()
        loss_gauss = adjaceny_matrix * self.gaussian_nll(pred_im, pred_logvar_im, target_mean) # TODO: This can lead to NaNs for the mean or the var of the entries that are not observed (adjacency_matrix = 0).
        loss_initial = self.init_cross_entropy(pred_init_cond, target_init_cond)

        loss_missing_link = normalization_constants * (1.0 - adjaceny_matrix) * (torch.pow(pred_im, 2) + torch.exp(pred_logvar_im))
        gaus_cons = schedulers.get("gauss_nll")(step) if schedulers else 1.0
        init_cons = schedulers.get("init_cross_entropy")(step) if schedulers else 1.0
        missing_link_cons = schedulers.get("missing_link")(step) if schedulers else 1.0
        loss = gaus_cons * loss_gauss + init_cons * loss_initial.view(-1, 1) - missing_link_cons * loss_missing_link
        return {
            "loss": loss.mean(),
            "loss_gauss": loss_gauss.mean(),
            "loss_initial": loss_initial.mean(),
            "loss_missing_link": loss_missing_link.mean(),
            "beta_gauss_nll": gaus_cons,
            "beta_init_cross_entropy": init_cons,
            "beta_missing_link": missing_link_cons,
        }

    def new_stats(self) -> dict: ...
    def metric(self, y: Any, y_target: Any) -> Dict:
        return super().metric(y, y_target)


ModelFactory.register("FIMMJP", FIMMJP)
