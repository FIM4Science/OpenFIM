import os
from pathlib import Path
import torch
from fim.utils.grids import (
    define_mesh_points,
    random_size_consecutive_locations
)

from typing import List,Tuple
from fim.data.datasets import (
    FIMSDEDatabatch
)

import yaml
from abc import ABC, abstractmethod

from fim.data.data_generation.dynamical_systems import (
    DynamicalSystem,
    DYNAMICAL_SYSTEM_TO_MODELS,
    DYNAMICS_LABELS
)

# ------------------------------------------------------------------------------------------
# INTEGRATORS

class SDEIntegrator(ABC):
    @abstractmethod
    def step(self, states, system, drift_params, diffusion_params):
        """Performs one integration step."""
        pass

class EulerMaruyama(SDEIntegrator):

    def __init__(self,integrator_params: dict):
        self.dt = integrator_params["time_step"]
        self.stochastic = integrator_params["stochastic"]

    def step(self, states, system, drift_params, diffusion_params):
        drift = system.drift(states, None, drift_params)
        diffusion = system.diffusion(states, None, diffusion_params)
        new_states = states + drift * self.dt
        if self.stochastic:
            new_states = new_states + diffusion * torch.sqrt(torch.tensor(self.dt)) * torch.randn_like(states)
        return new_states

INTERGRATORS_METHODS = {"EulerMaruyama":EulerMaruyama}

# ------------------------------------------------------------------------------------------
# PATH GENERATORS

class PathGenerator:
    """
    Class that generates data for the FIMPOODEDataBulk or FIMSDEpDataBulk according to the 
    observation parameters from ode_systems_hyperparams.yaml or poode_systems_hyperparams.yaml

    This is for a prescribed dynamical system, one data point consist of a parameter realization
    of one of the dynamical systems, each with a given number of paths and  observation times and 
    points.
    """
    system:DynamicalSystem

    def __init__(self, dataset_type:str,system: DynamicalSystem, integrator_params: dict):
        """
        Args:
            -dataset_type (str): which type of dataset will be used
            -system (DynamicalSystem): what to integrate
            -integrator parameters dict: parameters for the iterator
        """
        self.dataset_type = dataset_type
        self.system = system
        self.dt = integrator_params["time_step"]
        self.num_locations = integrator_params["num_locations"]

        self.num_realizations = self.system.num_realizations
        self.state_dim = self.system.state_dim

        # includes paths and realizations
        self.num_paths = integrator_params["num_paths"]
        self.total_num_paths = integrator_params["num_paths"]*self.num_realizations
        self.num_steps = integrator_params["num_steps"]
        self.stochastic = integrator_params["stochastic"]

        # observation parameters
        if "observation_time" in integrator_params.keys():
            self.observation_time_params = integrator_params["observation_time"]
        else:
            self.observation_time_params = False
        if "observation_coordinate" in integrator_params.keys():
            self.observation_coordinate_params = integrator_params["observation_coordinate"]
        else:
            self.observation_coordinate_params = False
        if "observation_noise" in integrator_params.keys():
            self.observation_noise_params = integrator_params["observation_noise"]
        else:
            self.observation_noise_params = False

        self.integrator = INTERGRATORS_METHODS[integrator_params["method"]](integrator_params)

    def generate_paths(self)->FIMSDEDatabatch:
        """
        generate paths such that every realization of the parameters
        has a num of paths

        Returns 
            DataBulk (FIMPOODEDataBulk|FIMSDEpDataBulk)
        """
        states = self.system.sample_initial_states(self.total_num_paths)

        drift_params = self.system.sample_drift_params(self.num_realizations) #[num_realizations,max_drift_params]
        diffusion_params = self.system.sample_diffusion_params(self.num_realizations)#[num_realizations,max_diffusion_params]

        # repeats according to the numbr of paths per parameter realization
        drift_params_repeated = torch.repeat_interleave(drift_params,self.num_paths,0)
        diffusion_params_repeated = torch.repeat_interleave(diffusion_params,self.num_paths,0)

        # paths
        hidden_paths = torch.zeros((self.total_num_paths, self.num_steps + 1, self.state_dim))
        hidden_paths[:, 0] = states.clone()

        #times
        hidden_times = torch.linspace(0.,self.num_steps*self.dt,self.num_steps+1)
        hidden_times = hidden_times[None,:].repeat(self.total_num_paths,1)#[total_num_paths,max_diffusion_params]

        #go trough iterator
        for step in range(self.num_steps):
            states = self.integrator.step(states, self.system, drift_params_repeated, diffusion_params_repeated)
            hidden_paths[:, step + 1] = states.clone()

        # Undo repeat interleave i.e. first shape corresponds to number of realizations (parameters are not repited)
        hidden_paths = hidden_paths.view(self.num_realizations,self.num_paths,self.num_steps+1,-1)
        hidden_times = hidden_times.view(self.num_realizations,self.num_paths,self.num_steps+1,-1)

        return self.define_bulk(hidden_paths,hidden_times,drift_params,diffusion_params)

    def define_bulk(self,hidden_paths,hidden_times,drift_params,diffusion_params)->FIMSDEDatabatch:
        """
        Evaluates cases for the different data bulks

        in the case of POODE we call the time observations as well as the coordinate observations
        as well as the noise to the observations
        """
        if self.dataset_type == "FIMSDEpDataset":
            return self.define_fim_sde_data(hidden_paths,hidden_times,drift_params,diffusion_params)
        elif self.dataset_type == "FIMPOODEDataset":

            obs_values,obs_times,obs_mask,obs_lenght = self.time_observations_and_mask(hidden_paths,hidden_times)
            noisy_obs_values = self.obs_noise(obs_values)
            o_values = self.coordinate_observation(obs_values)

            return self.define_fim_poode_data(noisy_obs_values,
                                              o_values,
                                              obs_values,
                                              obs_times,
                                              obs_mask,
                                              obs_lenght,
                                              hidden_paths,
                                              hidden_times,
                                              drift_params,
                                              diffusion_params)

    def define_fim_sde_data(
        self,
        obs_values,
        obs_times,
        drift_parameters,
        diffusion_parameters
    )->FIMSDEDatabatch:
        """
        Defines hyper cube and evaluates drift and diffusion there

        Args:
            data:  obs_values,obs_times,drift_parameters,diffusion_parameters
        """
        process_label = torch.full((obs_values.size(0),1),DYNAMICS_LABELS[self.system.name_str])
        process_dimension = torch.full((obs_values.size(0),1),self.system.state_dim)

        num_paths = obs_values.size(0)
        dimensions = obs_values.size(3)

        hypercube_locations = define_mesh_points(self.num_locations,dimensions)
        num_hypercube_points = hypercube_locations.size(0)
        hypercube_ = hypercube_locations.repeat((self.num_realizations,1))

        drift_params_ = drift_parameters.repeat_interleave(num_hypercube_points,0)
        diffusion_parameters_ = diffusion_parameters.repeat_interleave(num_hypercube_points,0)

        drift_at_hypercube = self.system.drift(hypercube_,None,drift_params_)
        drift_at_hypercube = drift_at_hypercube.reshape(num_paths,num_hypercube_points,dimensions)

        diffusion_at_hypercube = self.system.diffusion(hypercube_,None,diffusion_parameters_)
        diffusion_at_hypercube = diffusion_at_hypercube.reshape(num_paths,num_hypercube_points,dimensions)
        hypercube_ = hypercube_.reshape(num_paths,num_hypercube_points,dimensions)

        data = FIMSDEDatabatch(
            locations=hypercube_,
            obs_times=obs_times,
            obs_values=obs_values,
            diffusion_at_locations=diffusion_at_hypercube,
            drift_at_locations=drift_at_hypercube,
            #diffusion_parameters=diffusion_parameters,
            #drift_parameters=drift_parameters,
            #process_label=process_label,
            process_dimension=process_dimension
        )
        return data

    def time_observations_and_mask(
            self,
            hidden_paths,
            hidden_times
        ):
        """
        In the Iterator parameters:

        observation_time:
            observation_time_type: "consecutive" # consecutive,  
            size_distribution: "poisson"
            av_num_observations: 20
            low: 20
            high: 100

        if no observation_time_params is given the hidden_values are then observed 
        values

        Returns
        --------
        obs_values,obs_times,obs_mask,obs_lenght
        """
        if self.observation_time_params:
            if self.observation_time_params["observation_time_type"] == "consecutive":
                obs_values,obs_times,obs_mask,obs_lenght = random_size_consecutive_locations(hidden_paths,hidden_times,self.observation_time_params)
                return obs_values,obs_times,obs_mask,obs_lenght
        else:
            # if no observation parameters then returns same hidden values
            B,P,T,_ = hidden_times.shape
            return hidden_paths,hidden_times,torch.ones_like(hidden_times),torch.full((B,P),T)

    def coordinate_observation(self,obs_values):
        """Not implemented keeps first coordinates"""
        if self.observation_coordinate_params:
            return obs_values[:,:,:,0].unsqueeze(-1)
        else:
            return obs_values

    def obs_noise(self,obs_values):
        """Not implemented keeps the same values"""
        if self.obs_noise:
            return obs_values
        else:
            return obs_values

def set_up_a_dynamical_system(
        dataset_type:str,
        params_yaml:dict,
        integrator_params:dict,
        experiment_dir:str,
        return_data:bool=True,
    )->DynamicalSystem|FIMSDEDatabatch:
    """
    Takes a dict of parameters from yaml and creates
    the dynamical system model and generate the data accordingly
    every time the data is generated it will be saved and will only be 
    regenerated is so desided

    Args:
        -dataset_type (str): which type of dataset will be used
        -params_yaml (dict): dynamical system model parameters as dict
        -integrator_params: itegrator parameters 
        -experiment_dir (str): where all the models data is saved
        -return_data (bool): if true returns the FIMSDEpDataBulk otherwise the model
    
    Returns
        DynamicalSystem|FIMSDEpDataBulk
    """
    dynamical_name_str = params_yaml.get("name", "")
    study_name_str = params_yaml.get("data_bulk_name", "default")
    redo_study = params_yaml.get("redo", False)

    study_path = Path(os.path.join(experiment_dir,study_name_str+".tr"))

    if dynamical_name_str in DYNAMICAL_SYSTEM_TO_MODELS.keys():
        # Create an instance of OneCompartmentModelParams with the loaded values
        dynamical_model = DYNAMICAL_SYSTEM_TO_MODELS[dynamical_name_str](params_yaml)

    if return_data:
        data:FIMSDEDatabatch
        # study data does not exist we generated again
        if not study_path.exists():
            path_generator = PathGenerator(dataset_type,dynamical_model,integrator_params)
            data = path_generator.generate_paths()
            torch.save(data,study_path)
            return data
        else:
            # data exist but we must simulate again
            if redo_study:
                path_generator = PathGenerator(dataset_type,dynamical_model,integrator_params)
                data = path_generator.generate_paths()
                torch.save(data,study_path)
                return data
            # data exist and we take it
            else:
                data = torch.load(study_path)
                return data

    return dynamical_model

def define_dynamicals_models_from_yaml(
        yaml_file: str,
        return_data:bool=True,
    )->Tuple[str,
             List[DynamicalSystem|FIMSDEDatabatch],
             List[DynamicalSystem|FIMSDEDatabatch],
             List[DynamicalSystem|FIMSDEDatabatch]]:
    """
    Function to load or generate different studies from a yaml file,
    this is the function that will allow the dataloader to get the data
    from the dynamic simulations
    
    Args:
        yaml_file: str of yaml file that contains a list of hyper parameters 
        from different compartment models, one such hyperparameters allows the 
        the set_up_a_study function (above) to generate one population study

        return_data: bool if false will return the dynamic system if true 
                     will return the DataBulk object
    """
    from fim import data_path
    with open(yaml_file, 'r') as file:
        data = yaml.safe_load(file)

    # check the experiment folder exist
    experiment_name = data["experiment_name"]
    experiment_dir = os.path.join(data_path,experiment_name)
    if not os.path.exists(experiment_dir):
        # Create the folder
        os.makedirs(experiment_dir)

    # data type
    dataset_type = data["dataset_type"]
    # integrator params
    integrator_params = data["integration"]

    # generate the data
    train_studies:List[DynamicalSystem|FIMSDEDatabatch] = []
    test_studies:List[DynamicalSystem|FIMSDEDatabatch] = []
    validation_studies:List[DynamicalSystem|FIMSDEDatabatch] = []

    for params_yaml in data['train']:
        compartment_model = set_up_a_dynamical_system(dataset_type,params_yaml,integrator_params,experiment_dir,return_data)
        train_studies.append(compartment_model)

    for params_yaml in data['test']:
        compartment_model = set_up_a_dynamical_system(dataset_type,params_yaml,integrator_params,experiment_dir,return_data)
        test_studies.append(compartment_model)

    for params_yaml in data['validation']:
        compartment_model =  set_up_a_dynamical_system(dataset_type,params_yaml,integrator_params,experiment_dir,return_data)
        validation_studies.append(compartment_model)

    return (
        dataset_type,
        experiment_name,
        train_studies,
        test_studies,
        validation_studies
    )
