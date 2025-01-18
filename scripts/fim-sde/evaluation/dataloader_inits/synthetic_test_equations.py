from functools import partial
from pathlib import Path
from typing import Optional

from fim.utils.evaluation_sde_synthetic_datasets import get_synthetic_dataloader


def get_svise_dataloaders_inits(
    svise_dir: Optional[Path] = None, batch_size: Optional[int] = 32, num_workers: Optional[int] = 4
) -> tuple[dict]:
    """
    Return DataLoaderInitializer per svise test equation.

    Args:
        svise_dir (Path): Absolute path to dir with equations for subdirs.

    Returns:
        dataloder_dict, dataloader_map
    """
    if svise_dir is None:
        svise_dir = Path("/cephfs_projects/foundation_models/data/SDE/test/20241222_svise_1_perc_diffusion_no_additive_noise")

    _get_synthetic_dataloader = partial(get_synthetic_dataloader, batch_size=batch_size, num_workers=num_workers)

    dataloader_dict = {
        "svise_damped_cubic": _get_synthetic_dataloader([svise_dir / "damped_cubic_oscillator/"]),
        "svise_damped_linear": _get_synthetic_dataloader([svise_dir / "damped_linear_oscillator/"]),
        "svise_duffing": _get_synthetic_dataloader([svise_dir / "duffing_oscillator/"]),
        "svise_hopf_bifurcation": _get_synthetic_dataloader([svise_dir / "hopf_bifurcation/"]),
        "svise_lorenz": _get_synthetic_dataloader([svise_dir / "lorenz_63/"]),
        "svise_selkov_glycolysis": _get_synthetic_dataloader([svise_dir / "selkov_glycolysis/"]),
    }

    dataloader_display_ids = {
        "svise_damped_cubic": "SVISE: Damped Cubic Oscillator",
        "svise_damped_linear": "SVISE: Damped Linear Oscillator",
        "svise_duffing": "SVISE: Duffing Oscillator",
        "svise_hopf_bifurcation": "SVISE: Hopf Bifirucation",
        "svise_lorenz": "SVISE: Lorenz",
        "svise_selkov_glycolysis": "SVISE: Selkov Glycolysis",
    }

    return dataloader_dict, dataloader_display_ids


def get_up_to_deg_3_polynomial_test_sets_init(
    poly_dir: Optional[Path] = None, batch_size: Optional[int] = 32, num_workers: Optional[int] = 4
) -> tuple[dict]:
    """
    Return DataLoaderInitializer per polynomial test set

    Args:
        poly_dir (Path): Absolute path to dir with test sets for subdirs.

    Returns:
        dataloder_dict, dataloader_map
    """
    if poly_dir is None:
        poly_dir = Path("/cephfs_projects/foundation_models/data/SDE/test/20250105_polynomial_test_sets_100_each_50_paths")

    _get_synthetic_dataloader = partial(get_synthetic_dataloader, batch_size=batch_size, num_workers=num_workers)

    dataloader_dict = {
        "poly_deg_drift_1_deg_diff_0_std_normal_init": _get_synthetic_dataloader(
            [
                poly_dir / "dim_1_deg_drift_1_deg_diff_0_std_normal_init/",
                poly_dir / "dim_2_deg_drift_1_deg_diff_0_std_normal_init/",
                poly_dir / "dim_3_deg_drift_1_deg_diff_0_std_normal_init/",
            ]
        ),
        "poly_deg_drift_1_deg_diff_1_std_normal_init": _get_synthetic_dataloader(
            [
                poly_dir / "dim_1_deg_drift_1_deg_diff_1_std_normal_init/",
                poly_dir / "dim_1_deg_drift_1_deg_diff_1_std_normal_init/",
                poly_dir / "dim_1_deg_drift_1_deg_diff_1_std_normal_init/",
            ]
        ),
        "poly_deg_drift_2_deg_diff_0_std_normal_init": _get_synthetic_dataloader(
            [
                poly_dir / "dim_1_deg_drift_2_deg_diff_0_std_normal_init/",
                poly_dir / "dim_2_deg_drift_2_deg_diff_0_std_normal_init/",
                poly_dir / "dim_3_deg_drift_2_deg_diff_0_std_normal_init/",
            ]
        ),
        "poly_deg_drift_2_deg_diff_1_std_normal_init": _get_synthetic_dataloader(
            [
                poly_dir / "dim_1_deg_drift_2_deg_diff_1_std_normal_init/",
                poly_dir / "dim_2_deg_drift_2_deg_diff_1_std_normal_init/",
                poly_dir / "dim_3_deg_drift_2_deg_diff_1_std_normal_init/",
            ]
        ),
        "poly_deg_drift_3_deg_diff_0_std_normal_init": _get_synthetic_dataloader(
            [
                poly_dir / "dim_1_deg_drift_3_deg_diff_0_std_normal_init/",
                poly_dir / "dim_2_deg_drift_3_deg_diff_0_std_normal_init/",
                poly_dir / "dim_3_deg_drift_3_deg_diff_0_std_normal_init/",
            ]
        ),
        "poly_deg_drift_3_deg_diff_1_std_normal_init": _get_synthetic_dataloader(
            [
                poly_dir / "dim_1_deg_drift_3_deg_diff_1_std_normal_init/",
                poly_dir / "dim_2_deg_drift_3_deg_diff_1_std_normal_init/",
                poly_dir / "dim_3_deg_drift_3_deg_diff_1_std_normal_init/",
            ]
        ),
    }

    dataloader_display_ids = {
        "poly_deg_drift_1_deg_diff_0_std_normal_init": "Drift Deg. 1, Diff Deg. 0, Std. Normal. Init.",
        "poly_deg_drift_1_deg_diff_1_std_normal_init": "Drift Deg. 1, Diff Deg. 1, Std. Normal. Init.",
        "poly_deg_drift_1_deg_diff_2_std_normal_init": "Drift Deg. 1, Diff Deg. 2, Std. Normal. Init.",
        "poly_deg_drift_2_deg_diff_0_std_normal_init": "Drift Deg. 2, Diff Deg. 0, Std. Normal. Init.",
        "poly_deg_drift_2_deg_diff_1_std_normal_init": "Drift Deg. 2, Diff Deg. 1, Std. Normal. Init.",
        "poly_deg_drift_2_deg_diff_2_std_normal_init": "Drift Deg. 2, Diff Deg. 2, Std. Normal. Init.",
    }

    return dataloader_dict, dataloader_display_ids


def get_opper_or_wang_dataloaders_inits(
    opper_or_wang_dir: Optional[Path] = None, batch_size: Optional[int] = 32, num_workers: Optional[int] = 4
) -> tuple[dict]:
    """
    Return DataLoaderInitializer per opper or wang test equation.

    Args:
        opper_or_wang_dir (Path): Absolute path to dir with equations for subdirs.

    Returns:
        dataloder_dict, dataloader_map
    """
    if opper_or_wang_dir is None:
        opper_or_wang_dir = Path("/cephfs_projects/foundation_models/data/SDE/test/20241223_opper_and_wang_cut_to_128_lenght_paths")

    _get_synthetic_dataloader = partial(get_synthetic_dataloader, batch_size=batch_size, num_workers=num_workers)

    dataloader_dict = {
        "opper_double_well_constant_diff_5000_points": _get_synthetic_dataloader(
            [opper_or_wang_dir / "double_well_constant_diff_5000_points"]
        ),
        "opper_double_well_state_dep_diff_5000_points": _get_synthetic_dataloader(
            [opper_or_wang_dir / "double_well_state_dep_diff_5000_points"]
        ),
        "opper_two_d_10000_points": _get_synthetic_dataloader([opper_or_wang_dir / "two_d_opper_10000_points"]),
        "wang_two_d_80000_points": _get_synthetic_dataloader([opper_or_wang_dir / "two_d_wang_80000_points"]),
        "wang_double_well_25000_points": _get_synthetic_dataloader([opper_or_wang_dir / "double_well_wang_25000_points"]),
        "opper_lorenz_3000_points": _get_synthetic_dataloader([opper_or_wang_dir / "lorenz_3000_points"]),
    }

    dataloader_display_ids = {
        "opper_double_well_constant_diff_5000_points": "Opper: Double Well, Const. Diff., 5000 Points",
        "opper_double_well_state_dep_diff_5000_points": "Opper: Double Well, State Dep. Diff., 5000 Points",
        "opper_two_d_10000_points": "Opper: 2D Synth. System, Const. Diff., 10000 Points",
        "wang_two_d_80000_points": "Wang: 2D Synth. System, State Dep. Diff, 80000 Points",
        "wang_double_well_25000_points": "Wang: Double Well, State Dep. Diff., 25000 Points",
        "opper_lorenz_3000_points": "Opper: Lorenz, 3000 points",
    }

    return dataloader_dict, dataloader_display_ids


def get_cspd_dataloaders_inits(
    cspd_dir: Optional[Path] = None, batch_size: Optional[int] = 32, num_workers: Optional[int] = 4
) -> tuple[dict]:
    """
    Return DataLoaderInitializer per CSPD test equation.

    Args:
        cspd_dir (Path): Absolute path to dir with equations for subdirs.

    Returns:
        dataloder_dict, dataloader_map
    """
    if cspd_dir is None:
        cspd_dir = Path("/cephfs_projects/foundation_models/data/SDE/test/20250106_scdp_synthetic_datasets")

    _get_synthetic_dataloader = partial(get_synthetic_dataloader, batch_size=batch_size, num_workers=num_workers)

    dataloader_dict = {
        "cspd_cir_500_paths": _get_synthetic_dataloader([cspd_dir / "cir_500_paths"]),
        "cspd_lorenz_500_paths": _get_synthetic_dataloader([cspd_dir / "lorenz_500_paths"]),
        "cspd_lotka_volterra_500_paths": _get_synthetic_dataloader([cspd_dir / "lotka_volterra_500_paths"]),
        "cspd_orn_uhl_500_paths": _get_synthetic_dataloader([cspd_dir / "orn_uhl_500_paths"]),
        "cspd_sink_500_paths": _get_synthetic_dataloader([cspd_dir / "sink_500_paths"]),
    }

    dataloader_display_ids = {
        "cspd_cir_500_paths": "CSPD: CIR, 500 paths",
        "cspd_lorenz_500_paths": "CSPD: Lorenz, 500 paths",
        "cspd_lotka_volterra_500_paths": "CSPD: Lotka Volterra, 500 paths",
        "cspd_orn_uhl_500_paths": "CSPD: Ornstein Uhlenbeck, 500 paths",
        "cspd_sink_500_paths": "CSPD: Sink, 500 paths",
    }

    return dataloader_dict, dataloader_display_ids


def get_difficult_synth_equations_dataloaders_inits(
    data_path: Optional[Path] = None, batch_size: Optional[int] = 32, num_workers: Optional[int] = 4
) -> tuple[dict]:
    """
    Return DataLoaderInitializer per diffcult (non-polynomial) equation.

    Args:
        data dir (Path): Absolute path to dir with equations for subdirs.

    Returns:
        dataloder_dict, dataloader_map
    """
    if data_path is None:
        data_path = Path("/cephfs_projects/foundation_models/data/SDE/test/20250106_difficult_non_polynomial_equations")

    _get_synthetic_dataloader = partial(get_synthetic_dataloader, batch_size=batch_size, num_workers=num_workers)

    dataloader_dict = {
        "diff_exp_inv_300_paths": _get_synthetic_dataloader([data_path / "exp_-1_xx_300_paths"]),
        "diff_inverse_drift_300_paths": _get_synthetic_dataloader([data_path / "inverse_drift_300_paths"]),
        "diff_sign_diffusion_300_paths": _get_synthetic_dataloader([data_path / "sign_diffusion_300_paths"]),
        "diff_exp_inv_100_paths": _get_synthetic_dataloader([data_path / "exp_-1_xx_100_paths"]),
        "diff_inverse_drift_100_paths": _get_synthetic_dataloader([data_path / "inverse_drift_100_paths"]),
        "diff_sign_diffusion_100_paths": _get_synthetic_dataloader([data_path / "sign_diffusion_100_paths"]),
        "diff_exp_inv_50_paths": _get_synthetic_dataloader([data_path / "exp_-1_xx_50_paths"]),
        "diff_inverse_drift_50_paths": _get_synthetic_dataloader([data_path / "inverse_drift_50_paths"]),
        "diff_sign_diffusion_50_paths": _get_synthetic_dataloader([data_path / "sign_diffusion_50_paths"]),
    }

    dataloader_display_ids = {
        "diff_exp_inv_300_paths": "Difficult: Exp(-1/x^2) dt + dWt, 300 paths",
        "diff_inverse_drift_300_paths": "Difficult: -1/(2x) dt + dWt, 300 paths",
        "diff_sign_diffusion_300_paths": "Difficult: sgn(x) dWt, 300 paths",
        "diff_exp_inv_100_paths": "Difficult: Exp(-1/x^2) dt + dWt, 100 paths",
        "diff_inverse_drift_100_paths": "Difficult: -1/(2x) dt + dWt, 100 paths",
        "diff_sign_diffusion_100_paths": "Difficult: sgn(x) dWt, 100 paths",
        "diff_exp_inv_50_paths": "Difficult: Exp(-1/x^2) dt + dWt, 50 paths",
        "diff_inverse_drift_50_paths": "Difficult: -1/(2x) dt + dWt, 50 paths",
        "diff_sign_diffusion_50_paths": "Difficult: sgn(x) dWt, 50 paths",
    }

    return dataloader_dict, dataloader_display_ids
