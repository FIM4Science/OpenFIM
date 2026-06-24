import copy
import json
import math
import re
from pathlib import Path
from typing import Optional, OrderedDict, Tuple, List

import numpy as np
import sympy as sp
import torch
from jedi.inference.gradual.typing import Callable
try:
    from odeformer.model import SymbolicTransformerRegressor
except ImportError:
    SymbolicTransformerRegressor = None
from safetensors.torch import load_file
from transformers import PretrainedConfig

from fim.models.blocks import ModelFactory
from fim.models.ode import FIMODE
from fim.models.ode_trainer import FIMODEConfig as TrainingWrapperConfiguration
from fim.utils.helper import load_yaml


class PredictionModel:
    class PredictionModelError(RuntimeError):
        def __init__(self, cause, message="Prediction model had a problem"):
            self.message = message
            super().__init__(f"{message}, {cause}")

    def fit(self, traj: torch.Tensor, times: torch.Tensor, mask: Optional[torch.Tensor] = None):
        pass

    def system(self, location: torch.Tensor) -> torch.Tensor:
        """ Evaluates the predicted vector field (based on the context trajectories) at the given locations. """
        pass

    def is_fitted(self) -> bool:
        pass

    def get_model_identifier(self) -> str:
        return self.__class__.__name__


class OdeonEval(PredictionModel):
    """ This class actually loads model checkpoint and can predict vector field at arbitrary locations, over and over again. """

    model: FIMODE

    title: str
    epoch: int

    def __init__(self, path_to_checkpoints_dir: Path, epoch: Optional[int] = None):
        """ Loads the checkpoint etc. """

        config = load_yaml(path_to_checkpoints_dir / ".." / "train_parameters.yaml")
        self._model_path = path_to_checkpoints_dir
        self._loaded_config = config
        config = PretrainedConfig.from_dict(config)
        config.model_config = config.model["model_config"]
        config.train_config = config.model["train_config"]
        model_type = config.model["model_type"]

        if epoch is None:
            path = path_to_checkpoints_dir / "best-model"
        else:
            path = path_to_checkpoints_dir / f"epoch-{str(epoch)}"

        safetensors_path = path / "model.safetensors"
        pth_path = path / "model-checkpoint.pth"

        if safetensors_path.exists():
            weights = load_file(path / "model.safetensors")
        elif pth_path.exists():
            weights = torch.load(pth_path, map_location=torch.device('cpu'))
        else:
            raise FileNotFoundError(
                f"Model weights not found in {path}. Expected 'model.safetensors' or 'model-checkpoint.pth'.")

        class_training_wrapper = ModelFactory.model_types[model_type]
        training_wrapper = class_training_wrapper(config)
        try:
            missing_keys, unexpected_keys = training_wrapper.load_state_dict(weights, strict=True)
        except Exception as _:
            weights = self.map_old_to_new_state_dict(weights)
            missing_keys, unexpected_keys = training_wrapper.load_state_dict(weights, strict=True)

        assert len(missing_keys) == 0 and len(unexpected_keys) == 0
        self.model = training_wrapper.model
        self.model.eval()

        self.u_model = training_wrapper.u_model
        self.u_model.eval()

        train_state = torch.load(path / "train-state-checkpoint.pth", weights_only=False)
        self.epoch = train_state["last_epoch"]
        self.title = f"FIM-{self.epoch}-{train_state['params']['experiment']['name']}"

        print(self.title)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #device = torch.device("cpu")
        self.device = device
        self.model.to(device)
        self.u_model.to(device)

    def map_old_to_new_state_dict(self, weights):
        """ I guess this is backward compatibility function for loading old model checkpoints. """

        def map_old_to_new_keys(old_key):
            key = old_key.replace("context_encoder.layers", "trajectory_encoder.context_encoder.layers")
            if key.startswith("model.functional_encoder"):
                key = key.replace("model.functional_encoder", "model.functional_decoder")
            key = key.replace("model.delta_t_proj", "model.trajectory_encoder.delta_t_proj")
            key = key.replace("model.delta_x_proj", "model.trajectory_encoder.delta_x_proj")
            key = key.replace("model.x_proj", "model.trajectory_encoder.x_proj")
            key = key.replace("model.delta_x_squared_proj", "model.trajectory_encoder.delta_x_squared_proj")

            return key

        new_state_dict = OrderedDict()
        for old_key, value in weights.items():
            new_key = map_old_to_new_keys(old_key)
            new_state_dict[new_key] = value
        weights = new_state_dict
        return weights

    @torch.no_grad()
    def fit(self, traj: torch.Tensor, times: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        self.traj = traj    # (b,t,n,d)
        self.times = times
        self.mask = mask if mask is not None else torch.ones_like(self.traj[..., :1], dtype=torch.bool,
                                                                  device=traj.device)

        # cache the trajectory encoding: this requires changes to system() too...
        # self.traj_encoding = self.model.trajectory_encoding(self.traj, self.times, self.mask)
        
        return torch.arange(0, traj.shape[0], dtype=torch.long, device=traj.device)

    @torch.no_grad()
    def system(self, location: torch.Tensor) -> torch.Tensor:
        # out = self.model.forward(self.traj, self.times, location, self.mask, torch.ones((1, 1, 3), dtype=torch.bool))
        out = self.model.forward(self.traj, self.times, location, self.mask)
        res = self.model.get_prediction_for_eval(out)
        res = res[..., :location.shape[-1]]
        return res.detach()

    @torch.no_grad()
    def system_with_u(self, location: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Like system, but with uncertainty quantification. """
        
        out = self.model.forward(self.traj, self.times, location, self.mask)
        y = self.model.get_prediction_for_eval(out)
        y = y[..., :location.shape[-1]]

        u = self.u_model.forward(out)
        u = u.detach()

        return y, u

    def is_fitted(self) -> bool:
        return hasattr(self, "traj") and hasattr(self, "times") and hasattr(self, "mask")

    def get_model_identifier(self) -> str:
        return self.title



# Patch torch.load to default weights_only=False for ODEFormer compatibility
# This fixes the PyTorch 2.6+ default weights_only=True issue.
# Uses function-local scope so the captured original can never be overwritten
# by a subsequent import or reload, preventing recursive closure bugs.
def _apply_torch_load_patch():
    if getattr(torch.load, '_weights_only_patched', False):
        return
    _orig = torch.load
    def _patched(*args, **kwargs):
        kwargs.setdefault('weights_only', False)
        return _orig(*args, **kwargs)
    _patched._weights_only_patched = True
    torch.load = _patched
_apply_torch_load_patch()

class OdeFormerEval(PredictionModel):
    def __init__(self, device: str = "cuda"):
        dstr = SymbolicTransformerRegressor(from_pretrained=True)
        if device == "cuda":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif device == "cpu":
            self.device = torch.device("cpu")
        elif device == "mps":
            self.device = torch.device("mps")
        else:
            raise ValueError(f"Invalid ODEFormerdevice: {device}")
            
        self.dtype = torch.float64

        dstr.set_model_args({'beam_size': 50, 'beam_temperature': 0.1})
        """
        if torch.cuda.is_available():
            print("calling .cuda()")
            dstr.model.cuda()
        """
        dstr.model.to(self.device)
        dstr.model.eval()
        print("odeformer device (new)", next(dstr.model.parameters()).device, "device available", self.device)

        self.model: SymbolicTransformerRegressor = dstr

        self.f: Callable = None
        self.symbolic_predictions: Optional[List[str]] = None

        self._map = OdeFormerEval._torch_lambdify_map_for_odeformer()

    @staticmethod
    def _torch_lambdify_map_for_odeformer():
        # picked out the relevant operations from: list(self.model.model.decoder.id2word.values())

        return {
        # operators
            'Add': torch.add,
            'Mul': torch.mul,
            'Pow': torch.pow,
            'Sub': torch.sub,
            'Div': torch.div,
            
            # functions
            'abs': torch.abs,
            'add': torch.add,
            'arccos': torch.arccos,
            'arcsin': torch.arcsin,
            'arctan': torch.arctan,
            'cos': torch.cos,
            'div': torch.div,
            'exp': torch.exp,
            'id': lambda x: x,
            'inv': torch.reciprocal,  # 1/x
            'log': torch.log,
            'mul': torch.mul,
            'pow': torch.pow,
            'pow2': lambda x: torch.pow(x, 2),
            'pow3': lambda x: torch.pow(x, 3),
            'rand': torch.rand,
            'sin': torch.sin,
            'sqrt': torch.sqrt,
            'sub': torch.sub,
            # dont know what t is supposed to be
            # 't': torch.t,
            'tan': torch.tan,

            # constants
            'e': math.e,
            'euler_gamma': float(sp.EulerGamma.evalf()),
            'pi': math.pi,
        }

    def _create_callable_func(self, expr, symbols):
        if expr.free_symbols:
            f_impl = sp.lambdify(symbols, expr, modules=[self._map], use_imps=False)

            def wrapped(*args):
                return f_impl(*args)

            return wrapped

        val = float(expr)

        def f_numeric(*args):
            like = args[0]
            return torch.full_like(like, fill_value=val)

        return f_numeric

    def fit(self, traj: torch.Tensor, times: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        trs, tms = [], []
        traj = traj.to(dtype=self.dtype)
        times = times.squeeze(dim=-1).to(dtype=self.dtype)

        for b in range(traj.shape[0]):
            tr = traj[b].squeeze(dim=0)
            ti = times[b].flatten()

            if mask is not None:
                m = mask[b].squeeze(dim=0).squeeze(dim=-1).bool()
                tr = tr[m]
                ti = ti[m]

            trs.append(tr.cpu().numpy())
            tms.append(ti.cpu().numpy())

        #print(trs[0].shape)     # (6,50,3)
        #print(tms[0].shape)     # (300,)
        prediction = self.model.fit(trajectories=trs, times=tms)

        D = traj.size(-1)
        symbols = [sp.symbols(f"x_{i}") for i in range(D)]
        eqs = [str(prediction[i][0]) for i in range(len(prediction))]
        self.symbolic_predictions = eqs

        valid_predictions_at = []
        systems: List[List[Callable]] = []
        for i, eq in enumerate(eqs):
            funcs = []
            equation_per_dim = str(eq).split("|")

            if len(equation_per_dim) != D:
                # raise PredictionModel.PredictionModelError(f"Expected {D} predicted equations, but got {len(equation_per_dim)}: {eq}")
                print(f"Expected {D} predicted equations, but got {len(equation_per_dim)}: {eq}")
                continue

            vars_found = set(int(num) for num in re.findall(r"x_(\d+)", eq))
            if len(vars_found) > D:
                print(f"Expected variables x_0 to x_{D - 1}, but found more in predicted equation: {eq}")
                continue

            has_correct_vars = all(i < D for i in vars_found)
            if not has_correct_vars:
                print(f"Expected variables x_0 to x_{D - 1}, but found: {vars_found} in predicted equation: {eq}")
                continue

            valid_predictions_at.append(i)
            for e in equation_per_dim:
                f_exec = sp.sympify(e)
                f_exec = self._create_callable_func(f_exec, copy.deepcopy(symbols))
                funcs.append(f_exec)
            systems.append(funcs)

        def make_system(funcs):
            def system(yb: torch.Tensor) -> torch.Tensor:
                args = yb.T.unbind(0)
                outs = [f(*args) for f in funcs]
                return torch.stack(outs, dim=-1)

            return system

        self._systems = [make_system(system) for system in systems]

        def f(y: torch.Tensor) -> torch.Tensor:
            assert y.dim() == 3, "Expected y with shape [B, N, D]"
            B = y.shape[0]
            assert B == len(self._systems), "Batch D of input does not match number of systems."

            with torch.inference_mode():
                res = []
                for b in range(B):
                    out_b = self._systems[b](y[b])
                    res.append(out_b)
                return torch.stack(res, dim=0)

        self.f = f

        return torch.tensor(valid_predictions_at).long().to(traj.device)


    def system(self, location: torch.Tensor) -> torch.Tensor:
        return self.f(location)

    def is_fitted(self) -> bool:
        return self.f is not None

    def get_model_identifier(self) -> str:
        return "OdeFormerEval"


class FimOdeHFEval(PredictionModel):
    """FIMODE loaded from HuggingFace, wrapped in the PredictionModel interface.

    Context is pre-encoded once in ``fit()`` via ``trajectory_encoding``; only
    ``function_decoding`` is called at each integration step in ``system()``.
    """

    _REPO_ID   = "FIM4Science/fim-ode"
    _SUBFOLDER = "base_model/checkpoints/best-model"

    def __init__(self, device: str = "cpu"):
        from huggingface_hub import hf_hub_download

        config_path  = hf_hub_download(self._REPO_ID, f"{self._SUBFOLDER}/config.json")
        weights_path = hf_hub_download(self._REPO_ID, f"{self._SUBFOLDER}/model.safetensors")

        with open(config_path) as f:
            config_dict = json.load(f)
        config = TrainingWrapperConfiguration()
        config.model_config = config_dict["model_config"]
        config.train_config = config_dict["train_config"]

        fim_model = FIMODE(config)
        state_dict = load_file(weights_path, device=device)
        state_dict = {k.removeprefix("model."): v for k, v in state_dict.items()}
        fim_model.load_state_dict(state_dict)
        fim_model = fim_model.to(device)
        fim_model.eval()

        self._fim = fim_model
        self.device = torch.device(device)
        self._wrapped_D    = None
        self._feature_mask = None
        self._concept      = None
        self._D_in         = None

    @torch.no_grad()
    def fit(self, traj: torch.Tensor, times: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is None:
            mask = torch.ones(*traj.shape[:-1], 1, dtype=torch.bool, device=traj.device)
        self._D_in = traj.shape[-1]
        self._wrapped_D, self._feature_mask, self._concept = self._fim.trajectory_encoding(traj, times, mask)
        return torch.arange(traj.shape[0], dtype=torch.long, device=traj.device)

    @torch.no_grad()
    def system(self, location: torch.Tensor) -> torch.Tensor:
        loc      = self._fim.pad_if_necessary(location)
        loc_norm = self._fim.spatial_norm.normalization_map(loc, self._concept._states_norm_stats)
        out      = self._fim.function_decoding(loc_norm, self._feature_mask, self._wrapped_D, self._concept)
        return self._fim.get_prediction_for_eval(out)[..., :self._D_in]

    def is_fitted(self) -> bool:
        return self._wrapped_D is not None

    def get_model_identifier(self) -> str:
        return "FimOdeHFEval"


if __name__ == '__main__':
    odeon = OdeonEval(Path("models/base_model/checkpoints"))
    odeformer = OdeFormerEval()
