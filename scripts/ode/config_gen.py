##!/usr/bin/env -S uv run python

import copy
from pathlib import Path
from typing import Any, Union

import yaml  # Requires PyYAML


# Add support for !!python/tuple while keeping SafeLoader semantics.
class TupleSafeLoader(yaml.SafeLoader):
    pass


def _construct_python_tuple(loader: yaml.Loader, node: yaml.Node):
    return tuple(loader.construct_sequence(node))


TupleSafeLoader.add_constructor("tag:yaml.org,2002:python/tuple", _construct_python_tuple)


# Dumper to emit tuples as !!python/tuple
class TupleSafeDumper(yaml.SafeDumper):
    pass


def _represent_python_tuple(dumper: yaml.Dumper, data: tuple):
    return dumper.represent_sequence("tag:yaml.org,2002:python/tuple", list(data))


TupleSafeDumper.add_representer(tuple, _represent_python_tuple)


def load_yaml(path: Union[str, Path]) -> Any:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return yaml.load(f, Loader=TupleSafeLoader)


def save_yaml(data: Any, path: Union[str, Path]):
    p = Path(path)
    with p.open("w", encoding="utf-8") as f:
        yaml.dump(data, f, Dumper=TupleSafeDumper, sort_keys=False, allow_unicode=True, default_flow_style=False)


def create_configs(
    base_config: dict,
    train_path: Path,
    valid_path: Path,
    test_path: Path,
    task_name: str,
    max_num_points: int,
    traj_loss_steps=(10,),
    intermediate_steps_per_step=(1,),
    use_h_max=False,
    h_maxs=(0.01,),
    loss_types=("mse", "l1"),
    with_noise=False,
    ic_noise_scales=(0.0,),
    step_noise_scales=(0.0,),
    lr=1.0e-05,
):

    all_configs = []
    for loss_steps in traj_loss_steps:
        for intermediate_steps in intermediate_steps_per_step:
            for h_max in h_maxs:
                for loss_type in loss_types:
                    for ic_noise_scale in ic_noise_scales:
                        for step_noise_scale in step_noise_scales:
                            assert loss_steps < max_num_points, (
                                "Number of loss steps must be strictly less than observed trajectory length!"
                            )

                            experiment_name = task_name + f"_losssteps={loss_steps}"
                            experiment_name += f"_ninter={intermediate_steps}" if not use_h_max else f"_hmax={h_max}"
                            experiment_name += f"_icnoise={ic_noise_scale:.3f}_stepnoise={step_noise_scale:.3f}" if with_noise else ""
                            experiment_name += f"_loss={loss_type}"

                            config = copy.deepcopy(base_config)

                            config["experiment"]["name"] = experiment_name

                            config["model"]["train_config"]["num_ic"] = min(max_num_points - loss_steps, 50)  # maybe change?
                            config["model"]["train_config"]["traj_loss_steps"] = loss_steps

                            config["model"]["train_config"]["use_h_max"] = use_h_max
                            if use_h_max:
                                config["model"]["train_config"]["h_max"] = h_max
                            else:
                                config["model"]["train_config"]["intermediate_steps_per_step"] = intermediate_steps

                            config["model"]["train_config"]["ic_noise_scale"] = ic_noise_scale if with_noise else 0.0
                            config["model"]["train_config"]["step_noise_scale"] = step_noise_scale if with_noise else 0.0

                            config["model"]["train_config"]["loss_type"] = loss_type

                            config["model"]["train_config"]["integrator_for_trajectory_training"] = "improved_euler"

                            config["dataset"]["data_dirs"]["train"] = (str(train_path),)
                            config["dataset"]["data_dirs"]["validation"] = (str(valid_path),)  # unfortunately, this can't be an empty tuple
                            config["dataset"]["data_dirs"]["test"] = (str(test_path),) if test_path is not None else ()

                            config["trainer"]["experiment_dir"] += task_name + "/" + experiment_name

                            config["trainer"]["learning_rate"] = lr

                            all_configs.append(config)

    return all_configs


def generate_configs_for_different_noises():
    """Irrelevant now that we're using the Hegde et al. data"""

    path_to_base_config = Path("experiments/base_config_for_finetuning.yaml")
    base_config = load_yaml(path_to_base_config)

    grid_search_vp1 = {
        "intermediate_steps_per_step": (2,),  # (1,2,4,8,16),
        "traj_loss_steps": (49,),  # (3,6,10,15,25,45),
        "with_noise": False,
        "loss_types": ("mse",),
    }

    data_path = Path("experiments/vdp1/data")
    all_configs_vp1 = []
    for i in range(10):
        all_configs_vp1 += create_configs(
            base_config,
            train_path=data_path / f"{i}",
            valid_path=data_path / f"{i}",
            test_path=None,
            task_name="vdp1",
            max_num_points=50,
            **grid_search_vp1,
        )
    for i, config in enumerate(all_configs_vp1):
        configs_save_path = Path(f"experiments/vdp1/configs/config_noise_{i}.yaml")
        configs_save_path.parent.mkdir(parents=True, exist_ok=True)
        save_yaml(config, configs_save_path)

    print("Generated configs for VDP1.")

    grid_search_vp2 = {
        "use_h_max": True,
        "h_maxs": (0.025,),  # (7./50., 7./100., 7./200., 7./400.),
        "traj_loss_steps": (49,),  # (3,6,10,15,25,45),
        "with_noise": False,
        "loss_types": ("mse",),
    }

    data_path = Path("experiments/vdp2/data")
    all_configs_vp2 = []
    for i in range(10):
        all_configs_vp2 += create_configs(
            base_config,
            train_path=data_path / f"{i}",
            valid_path=data_path / f"{i}",
            test_path=None,
            task_name="vdp2",
            max_num_points=50,
            **grid_search_vp2,
        )
    for i, config in enumerate(all_configs_vp2):
        configs_save_path = Path(f"experiments/vdp2/configs/config_noise_{i}.yaml")
        configs_save_path.parent.mkdir(parents=True, exist_ok=True)
        save_yaml(config, configs_save_path)

    print("Generated configs for VDP2.")

    grid_search_fhn = {
        "use_h_max": True,
        "h_maxs": (0.001,),  # (.2, .1, .05, .025, .0125),
        # "intermediate_steps_per_step": (30,),
        "traj_loss_steps": (3,),  # (3,5,7,9,11,14,18),
        "with_noise": False,
        "loss_types": ("mse",),
        "lr": 1.0e-5,
    }

    data_path = Path("experiments/fhn/data")
    all_configs_fhn = []
    for i in range(10):
        all_configs_fhn += create_configs(
            base_config,
            train_path=data_path / f"{i}",
            valid_path=data_path / f"{i}",
            test_path=None,
            task_name="fhn",
            max_num_points=19,
            **grid_search_fhn,
        )
    for i, config in enumerate(all_configs_fhn):
        configs_save_path = Path(f"experiments/fhn/configs/config_noise_{i}.yaml")
        configs_save_path.parent.mkdir(parents=True, exist_ok=True)
        save_yaml(config, configs_save_path)

    print("Generated configs for FHN.")


def config_gen():
    path_to_base_config = Path("experiments/base_config_for_finetuning.yaml")
    base_config = load_yaml(path_to_base_config)
    # print(json.dumps(base_config, indent=2, ensure_ascii=False))

    grid_search_vp1 = {
        "intermediate_steps_per_step": (16,),  # (1,2,4,8,16),
        "traj_loss_steps": (15,),  # (3,6,10,15,25,45),
        "with_noise": False,
    }
    data_path = Path("experiments/vdp1/data/0")
    all_configs_vp1 = []
    for i in range(10):
        all_configs_vp1 += create_configs(
            base_config,
            train_path=data_path / f"{i}",
            valid_path=data_path,
            test_path=None,
            task_name="vdp1",
            max_num_points=50,
            **grid_search_vp1,
        )
    for i, config in enumerate(all_configs_vp1):
        configs_save_path = Path(f"experiments/vdp1/configs/config_noise_{i}.yaml")
        configs_save_path.parent.mkdir(parents=True, exist_ok=True)
        save_yaml(config, configs_save_path)

    for i, config in enumerate(all_configs_vp1):
        configs_save_path = Path(f"experiments/vdp1/configs/config_{i + 1:02d}.yaml")
        configs_save_path.parent.mkdir(parents=True, exist_ok=True)
        save_yaml(config, configs_save_path)

    grid_search_vp2 = {
        "use_h_max": True,
        "h_maxs": (7.0 / 400.0),  # (7./50., 7./100., 7./200., 7./400.),
        "traj_loss_steps": (15,),  # (3,6,10,15,25,45),
        "with_noise": False,
    }
    data_path = Path("experiments/vdp2/data/0")
    all_configs_vp2 = create_configs(
        base_config, train_path=data_path, valid_path=data_path, test_path=None, task_name="vdp2", max_num_points=50, **grid_search_vp2
    )
    for i, config in enumerate(all_configs_vp2):
        configs_save_path = Path(f"experiments/vdp2/configs/config_{i + 1:02d}.yaml")
        configs_save_path.parent.mkdir(parents=True, exist_ok=True)
        save_yaml(config, configs_save_path)

    grid_search_fhn = {
        "use_h_max": True,
        "h_maxs": (0.0125,),  # (.2, .1, .05, .025, .0125),
        "traj_loss_steps": (11,),  # (3,5,7,9,11,14,18),
        "with_noise": False,
    }
    data_path = Path("experiments/fhn/data/0")
    all_configs_fhn = create_configs(
        base_config, train_path=data_path, valid_path=data_path, test_path=None, task_name="fhn", max_num_points=19, **grid_search_fhn
    )
    for i, config in enumerate(all_configs_fhn):
        configs_save_path = Path(f"experiments/fhn/configs/config_{i + 1:02d}.yaml")
        configs_save_path.parent.mkdir(parents=True, exist_ok=True)
        save_yaml(config, configs_save_path)

    # MoCap experiments
    mocap_experiments = {
        "mocap09short": {
            "max_num_points": 50,
            "traj_loss_steps": (10, 25, 40),
            "intermediate_steps_per_step": (1, 2, 5),
            "ic_noise_scales": (0.01, 0.05, 0.25),
            "step_noise_scales": (0.1, 0.35, 1.0, 2.0),
            "loss_types": ("mse", "l1"),
        },
        "mocap09long": {
            "max_num_points": 100,
            "traj_loss_steps": (10, 25, 50),
            "intermediate_steps_per_step": (1, 2, 5),
            "ic_noise_scales": (0.01, 0.05, 0.25),
            "step_noise_scales": (0.1, 0.35, 1.0, 2.0),
            "loss_types": ("mse", "l1"),
        },
        "mocap35short": {
            "max_num_points": 50,
            "traj_loss_steps": (10, 25, 40),
            "intermediate_steps_per_step": (1, 2, 5),
            "ic_noise_scales": (0.01, 0.05, 0.25),
            "step_noise_scales": (0.35,),  # (0.1, 0.35, 1.0, 2.0),
            "loss_types": ("mse", "l1"),
        },
        "mocap35long": {
            "max_num_points": 250,
            "traj_loss_steps": (10, 25, 50, 100),
            "intermediate_steps_per_step": (1, 2, 5),
            "ic_noise_scales": (0.01, 0.05, 0.25),
            "step_noise_scales": (0.1, 0.35, 1.0, 2.0),
            "loss_types": ("mse", "l1"),
        },
        "mocap39short": {
            "max_num_points": 100,
            "traj_loss_steps": (10, 25, 50),
            "intermediate_steps_per_step": (1, 2, 5),
            "ic_noise_scales": (0.01, 0.05, 0.1, 0.25),
            "step_noise_scales": (0.1, 0.35, 1.0, 2.0),
            "loss_types": ("mse", "l1"),
        },
        "mocap39long": {
            "max_num_points": 250,
            "traj_loss_steps": (10, 25, 50, 100),
            "intermediate_steps_per_step": (1, 2, 5),
            "ic_noise_scales": (0.01, 0.05, 0.1, 0.25),
            "step_noise_scales": (0.1, 0.35, 1.0, 2.0),
            "loss_types": ("mse", "l1"),
        },
    }

    for task_name, experiment_dict in mocap_experiments.items():
        data_path = Path("experiments/mocap") / task_name / "data"
        train_path = data_path / "train" / "3d+2d" / "3d"
        valid_path = data_path / "valid" / "3d+2d" / "3d"
        test_path = data_path / "test" / "3d+2d" / "3d"

        all_configs = create_configs(
            base_config, train_path=train_path, valid_path=valid_path, test_path=test_path, task_name=task_name, **experiment_dict
        )
        for i, config in enumerate(all_configs):
            configs_save_path = Path(f"experiments/mocap/{task_name}/configs/config_{i + 1:02d}.yaml")
            configs_save_path.parent.mkdir(parents=True, exist_ok=True)
            save_yaml(config, configs_save_path)


if __name__ == "__main__":
    # generate_configs_for_different_noises()
    config_gen()
