import os
from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor
from tqdm import tqdm  # Import tqdm for the progress bar

from fim.data.datasets import FIMSDEDatabatch, FIMSDEDatabatchTuple
from fim.models.config_dataclasses import FIMSDEConfig
from fim.utils.helper import check_model_devices, nametuple_to_device


@dataclass
class FIMSDEPipelineOutput:
    locations: Tensor
    drift_at_locations_estimator: Tensor
    diffusion_at_locations_estimator: Tensor
    path: Tensor
    time: Tensor


class FIMSDEPipeline:
    """
    This pipeline follows the Huggingface transformers specs

    Inference Pipeline For SDE
    """

    config: FIMSDEConfig
    model: torch.nn

    def __init__(
        self,
        model: str,
    ):
        """
        Args:
            model (FIMSDEp,string)
                if string it should download a predefined transformer model
                it takes its parameters from the model
        """
        self.model = model
        self.config = model.model_config

        self.num_steps = self.config.number_of_time_steps_pipeline
        self.dt = self.config.dt_pipeline
        self.device = check_model_devices(model)

    def preprocess(self, databatch):
        """sent databatch to device of model"""
        databatch = nametuple_to_device(databatch, self.device)
        return databatch

    def _evaluate_at_grid(self, databatch: FIMSDEDatabatchTuple, locations=None):
        """ """
        if locations is None:
            locations = databatch.locations
        estimated_concepts = self.model(databatch, locations, training=False)
        return estimated_concepts.drift, estimated_concepts.diffusion

    def __call__(
        self,
        databatch: FIMSDEDatabatchTuple | FIMSDEDatabatch,
        initial_states: Tensor = None,
        evaluate_paths: bool = True,
        evaluate_at_locations: bool = True,
        locations: Tensor = None,
    ):
        self.model.eval()

        with torch.no_grad():
            databatch = self.preprocess(databatch)  # sent to device

            # evaluate paths
            if evaluate_paths:
                paths, times = self.model_euler_maruyama_loop(databatch, initial_states)
            else:
                paths, times = None, None

            # evaluate grids
            if evaluate_at_locations:
                drift_at_locations, diffusion_at_locations = self._evaluate_at_grid(databatch, locations)
            else:
                drift_at_locations, diffusion_at_locations = None, None

            # returns
            return FIMSDEPipelineOutput(
                locations=databatch.locations,
                drift_at_locations_estimator=drift_at_locations,
                diffusion_at_locations_estimator=diffusion_at_locations,
                path=paths,
                time=times,
            )

    def postprocess(self, model_outputs):
        pass

    # -------------------------- SAMPLES ------------------------------------------
    def sample_initial_states(
        self,
        databatch: FIMSDEDatabatchTuple,
    ) -> Tensor:
        # Initialize states for all paths
        dimensions = databatch.obs_values.size(3)
        num_paths = databatch.obs_values.size(0)
        states = torch.nn.functional.sigmoid(torch.normal(0.0, 1.0, size=(num_paths, dimensions), device=self.device))
        return states

    def model_as_drift_n_diffusion(self, X: Tensor, time: Tensor = None, databatch: FIMSDEDatabatchTuple = None) -> Tuple[Tensor, Tensor]:
        """
        Defines the drift and the diffusion from the forward pass
        and handles the padding accordingly

        Args:
            X (Tensor[B,D]): state
            time: (None)
            databatch (FIMSDEpDatabatchTuple):
        Returns:
            drift,diffusion [B,D]
        """
        X = X.unsqueeze(1)

        # Create a mask based on the dimensions
        dimension_mask = databatch.dimension_mask[:, 0, :]  # [B,D]

        # Get concepts at X
        estimated_concepts = self.model(databatch, X, training=False)

        # Apply the mask to X and Remove the Grid Shape
        drift = estimated_concepts.drift.squeeze() * dimension_mask.float()  # Zero out elements where mask is False
        diffusion = estimated_concepts.diffusion.squeeze() * dimension_mask.float()  # Zero out elements where mask is False

        return drift, diffusion

    def model_euler_maruyama_step(self, states: Tensor, databatch: FIMSDEDatabatchTuple) -> Tensor:
        """
        Assumes diagonal diffusion

        Args:
            states (Tensor[B,D])
            dt (float)
            model (FIMSDEp)
            databatch (databatch)
        Returns:
            new_states(Tensor[B,D])
        """
        # Calculate the deterministic part
        drift, diffusion = self.model_as_drift_n_diffusion(states, None, databatch)
        # Update the state with the deterministic part
        new_states = states + drift * self.dt
        # Add the diffusion part
        new_states += diffusion * torch.sqrt(torch.tensor(self.dt)) * torch.randn_like(states)
        return new_states

    def model_euler_maruyama_loop(
        self,
        databatch: FIMSDEDatabatchTuple = None,
        initial_states: Tensor = None,
    ):
        """
        Simulates paths from the Model using the Euler-Maruyama method.

        This is highly costly as the method needs to calculate a forward pass
        per Euler Mayorama Step, similar cost to what one will expect in a
        diffusion model.

        Args:
            num_steps: int = 100,
            dt: float = 0.01,
            model: FIMSDEp = None,
            databatch: FIMSDEpDatabatchTuple = None,
        Returns:
            paths(Tensor[B,number_of_steps,D]),times([B,number_of_steps,D])

        """
        dimensions = databatch.obs_values.size(3)
        num_paths = databatch.obs_values.size(0)

        # Initial states
        if initial_states is None:
            states = self.sample_initial_states(databatch)
        else:
            states = initial_states

        # Store paths
        paths = torch.zeros((num_paths, self.num_steps + 1, dimensions), device=self.device)  # +1 for initial state
        paths[:, 0] = states.clone()  # Store initial states

        times = torch.linspace(0.0, self.num_steps * self.dt, self.num_steps + 1, device=self.device)
        times = times[None, :].repeat(num_paths, 1)

        # Simulate the paths with tqdm progress bar
        for step in tqdm(range(self.num_steps), desc="Simulating steps", unit="step"):
            states = self.model_euler_maruyama_step(states, databatch)  # Diffusion term
            paths[:, step + 1] = states.clone()  # Store new states
        return paths, times


def sample_and_save_from_test(model, dataloaders, experiment_files):
    pipeline = FIMSDEPipeline(model)
    for batch_id, test_databatch in enumerate(dataloaders.test_it):
        test_output = pipeline(test_databatch)
        torch.save(test_output, os.path.join(experiment_files.sample_dir, "output_test_batch{0}.tr".format(batch_id)))
