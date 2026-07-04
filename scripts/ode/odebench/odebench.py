import json
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Final, List, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib import gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from odebench.stat_calculator import R2VarianceWeighterStatCalculator
from utils.eval_models import OdeFormerEval, OdeonEval, PredictionModel

from fim.data_generation.sde.lipschitz_systems import solve_ivp_one_step_method_with_delta_times
from fim.models.ode_trainer import DataCorruptionModel
from fim.models.ode_trainer import FIMODETrainingConfig as TrainingConfig


@dataclass(kw_only=True)
class OdeBenchConfig:
    model_type: Literal["fimodeon", "odeformer"]
    model_path: Optional[Path] = None
    model_checkpoint: Optional[int] = None
    max_dim: Optional[int] = None
    subsample_rate: Optional[float] = None
    noise_level: Optional[float] = None
    test_type: Literal["reconstruction", "generalization"]

    _max_systems: Optional[int] = None


@dataclass(kw_only=True)
class OdeData:
    @dataclass(kw_only=True)
    class Data:
        trajectories: Optional[torch.Tensor] = None
        times: Optional[torch.Tensor] = None

        def get_initial_conditions(self) -> torch.Tensor:
            return self.trajectories[:, 0, :]

    name: Optional[str] = "PLEASE SET NAME"
    model_name: Optional[str] = "PLEASE SET MODEL NAME"

    truth: Data

    metadata: Dict = None


class OdeBench:
    """Used to evaluate both FIMOdeon and ODEFormer on the ODEBench dataset"""

    MAX_POINTS_ODE_BENCH: Final[int] = 200

    def __init__(self, data_path: Path, results_base_path: Optional[Path] = None, device: str = "cuda"):
        self.data_path = data_path
        self.results_base_path = results_base_path or Path("experiments/odebench/results")

        torch.manual_seed(42)

        self.r2_calculator = R2VarianceWeighterStatCalculator()

        self.device = device
        if device == "cuda":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif device == "cpu":
            self.device = torch.device("cpu")
        else:
            raise ValueError(f"Invalid device: {device}")

        # plt.rcParams['font.family'] = 'TeX Gyre Pagella'
        # plt.rcParams.update({
        #     "text.usetex": True,
        #     "text.latex.preamble": r"\usepackage{amsmath}",
        # })

        self.palette = sns.color_palette()

        # Store current run folder for saving results
        self.current_run_folder = None

    def get_ode_bench_trajectories(self, max_dim: int = 3) -> List[OdeData]:
        device = self.device

        with open(self.data_path / "strogatz_extended.json", "r") as file:
            data_list = json.load(file)

        data: List[OdeData] = []
        for item in data_list:
            if item["dim"] > max_dim:
                continue

            equation_str = item["eq"]
            dim = item["dim"]
            metadata = {
                "equation": equation_str,
                "dim": dim,
            }

            for sol in item["solutions"][0]:
                assert sol["success"]
                constants = sol["consts"]
                metadata["constants"] = constants

                traj = torch.tensor(sol["y"]).T.unsqueeze(0)  # (T,D) -> (1,T,D)
                time = torch.tensor(sol["t"]).unsqueeze(dim=-1).unsqueeze(0)  # (T,)  -> (1,T,1)

                idxs = torch.linspace(
                    0, time.shape[1] - 1, self.MAX_POINTS_ODE_BENCH
                ).long()  # time.shape[1] is always 512, more than MAX_POINTS_ODE_BENCH
                traj = traj[:, idxs, :].to(device)  # (1,T,D) -> (1,MAX_POINTS_ODE_BENCH,D)
                time = time[:, idxs, :].to(device)  # (1,T,1) -> (1,MAX_POINTS_ODE_BENCH,1)

                d = OdeData(truth=OdeData.Data(trajectories=traj, times=time), name=item["eq_description"], metadata=metadata)

                data.append(d)

        return data

    def get_data_corruption_model(self, config: OdeBenchConfig) -> DataCorruptionModel:
        corruption_config = TrainingConfig(
            train_with_normalized_head=None,
            loss_filter_nans=None,
            loss_type=None,
            corruption_model_type="eval_odeformer",
            max_sigma_trajectory_noise=config.noise_level,
            max_subsampling_ration=config.subsample_rate,
        )

        data_corruption = DataCorruptionModel(corruption_config)

        return data_corruption

    def provide_data(self, config: OdeBenchConfig) -> Dict[int, Dict[str, torch.Tensor | List[str]]]:
        data = self.get_ode_bench_trajectories(config.max_dim)

        ode_per_dim = {i: {"tr": [], "tr_corrupted": [], "ti": [], "na": [], "mask": None} for i in range(1, config.max_dim + 1)}

        for item in data:
            dim_data = ode_per_dim.get(item.metadata["dim"])
            tr_dim = dim_data["tr"]
            ti_dim = dim_data["ti"]
            na_dim = dim_data["na"]  # list of equation names/descriptions

            tr = item.truth.trajectories
            ti = item.truth.times
            na = item.name

            tr_dim.append(tr)
            ti_dim.append(ti)
            na_dim.append(na)

        corruption_model = self.get_data_corruption_model(config)

        for dim, ode_data in ode_per_dim.items():
            traj = torch.cat(ode_data["tr"], dim=0).unsqueeze(dim=1)
            ode_data["tr"] = traj

            corrupted_data = corruption_model.corrupt_data(traj)
            ode_data["tr_corrupted"] = corrupted_data.corrupt_trajectory
            ode_data["mask"] = corrupted_data.mask

            ode_data["ti"] = torch.cat(ode_data["ti"], dim=0).unsqueeze(dim=1)
            ode_data["na"] = [
                n if len(n) <= 35 else n[:35] + "..." for n in ode_data["na"]
            ]  # truncates equation names/descriptions to 35 characters for display

        return ode_per_dim

    def load_model(self, config: OdeBenchConfig) -> OdeonEval | OdeFormerEval:
        if config.model_type == "odeformer":
            # dstr = SymbolicTransformerRegressor(from_pretrained=True)
            # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # model_args = {'beam_size': 50, 'beam_temperature': 0.1}
            # dstr.set_model_args(model_args)
            # dstr.model.to(device)
            # dstr.model.eval()

            # return dstr
            model = OdeFormerEval(device=self.device.type)
            return model

        if config.model_type == "fimodeon":
            assert config.model_path is not None
            model = OdeonEval(config.model_path, config.model_checkpoint)

            return model

        raise ValueError(f"Unknown model type: {config.model_type}. Supported types are 'fimodeon' and 'odeformer'.")

    def create_f(
        self, config: OdeBenchConfig, trajs: torch.Tensor, times: torch.Tensor, mask: torch.Tensor, names: List[str], model: PredictionModel
    ) -> Callable:
        """
        predicts the vector field from the observed trajectory
        """

        if config.model_type == "fimodeon":
            model.fit(traj=trajs, times=times, mask=mask)

            return model.system

        if config.model_type == "odeformer":
            model.fit(traj=trajs, times=times, mask=mask)

            # not really nice placement, but works for now
            n_items = len(names)
            fig_height = 0.8 * n_items  # adjust vertical space per item
            fig = plt.figure(figsize=(15, fig_height))
            gs = gridspec.GridSpec(n_items, 1, figure=fig, hspace=0.5)

            for i, (name, eq) in enumerate(zip(names, model.symbolic_predictions)):
                ax = fig.add_subplot(gs[i])
                ax.axis("off")  # hide axes
                ax.text(0, 1, f"{name}\n{eq}", fontsize=11, va="top")

            return model.system

        raise ValueError(f"Unknown model type: {config.model_type}")

    def do_test(
        self, ode_per_dim: Dict[int, torch.Tensor | str], model: PredictionModel, config: OdeBenchConfig
    ) -> Dict[int, Dict[str, torch.Tensor | List[str]]]:
        res = {}

        for dim in range(1, config.max_dim + 1):
            data_dim = ode_per_dim[dim]

            trajs = data_dim["tr"]
            corrupted_traj = data_dim["tr_corrupted"]
            mask = data_dim["mask"]
            times = data_dim["ti"]
            names = data_dim["na"]

            if config.test_type == "generalization":
                corrupted_traj = corrupted_traj.view(-1, 2, *corrupted_traj.shape[1:]).flip(dims=[1]).flatten(0, 1)
                times = times.view(-1, 2, *times.shape[1:]).flip(dims=[1]).flatten(0, 1)

            f = self.create_f(config, corrupted_traj, times, mask, names, model)

            y0 = trajs[:, :, 0, :]
            t0 = times[:, :, 0, :]
            delta_times = torch.diff(times, dim=2)
            ys, ts = solve_ivp_one_step_method_with_delta_times(y0, t0, delta_times, 8, f)

            ys = torch.cat(ys, dim=1)
            ts = torch.cat(ts, dim=1)
            trajs = trajs.squeeze(dim=1)

            if config.test_type == "generalization":
                corrupted_traj = corrupted_traj.view(-1, 2, *corrupted_traj.shape[1:]).flip(dims=[1]).flatten(0, 1)
                times = times.view(-1, 2, *times.shape[1:]).flip(dims=[1]).flatten(0, 1)

            corrupted_traj = corrupted_traj.squeeze(dim=1)
            mask = mask.squeeze(dim=1)
            times = times.squeeze(dim=1)
            self.plot_trajectories(ys, trajs, ts, corrupted_traj, mask, times, names)

            res[dim] = {"tr": trajs.cpu(), "ys": ys.cpu(), "ts": ts.cpu(), "mask": mask.cpu(), "times": times.cpu(), "names": names}

        return res

    def r2_score(self, y_true: torch.Tensor, y_pred: torch.Tensor, variance_weighted: bool = True) -> torch.Tensor:
        return self.r2_calculator.r2_score(y_true, y_pred, variance_weighted)

    def plot_trajectories(
        self,
        pred_traj: torch.Tensor,
        truth_traj: torch.Tensor,
        pred_times: torch.Tensor,
        input_traj: torch.Tensor,
        input_mask: torch.Tensor,
        input_times: torch.Tensor,
        names: List[str],
    ):
        """
        creates a visualization comparing predicted vs. ground truth trajectories for multiple ODE systems
        """

        # truth_traj: (B, T, D)

        num_plots = pred_traj.shape[0]
        cols = 2
        rows = (num_plots + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 8, rows * 6))
        axes = axes.flatten()

        r2_scores = self.r2_score(truth_traj, pred_traj, variance_weighted=False)

        for i in range(num_plots):
            ax = axes[i]

            truth_tr = truth_traj[i].cpu().numpy()
            pred_tr = pred_traj[i].cpu().numpy()
            pred_ts = pred_times[i].cpu().numpy()[:, 0]
            r2 = r2_scores[i].cpu().numpy()

            mask = input_mask[i, :, 0]
            num_non_maked = mask.sum().item()
            input_tr = input_traj[i][mask].reshape(num_non_maked, -1).cpu().numpy()
            input_ts_masked = input_times[i][mask].reshape(num_non_maked, -1).cpu().numpy()
            input_ts = input_times[i].cpu().numpy()

            for dim in range(truth_tr.shape[1]):
                ax.scatter(input_ts_masked[:, 0], input_tr[:, dim], alpha=0.2, color=self.palette[dim])
                ax.plot(
                    input_ts[:, 0],
                    truth_tr[:, dim],
                    linewidth=13,
                    alpha=0.15,
                    color=self.palette[dim],
                )

                ax.plot(pred_ts, pred_tr[:, dim], linewidth=6, alpha=0.4, color="white")
                ax.plot(pred_ts, pred_tr[:, dim], linewidth=3, alpha=0.9, label=f"Prediction $R^2$: {r2[dim]:.2f}", color=self.palette[dim])

            ax.set_title(names[i])
            ax.legend()
            ax.tick_params(axis="y", which="both", left=False, labelleft=False)
            ax.set_xlabel("Time $t$")

        # Remove unused subplots
        for j in range(num_plots, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        return fig

    def plot_r2_distribution(self, r2s: List[float], prefix: str = ""):
        """
        creates a histogram of R2 scores for a provided list of scores
        """

        fig, ax = plt.subplots(1, 1, figsize=(15, 10))

        data = np.array(r2s)
        data = np.nan_to_num(data, nan=-1000.0, neginf=-1000.0)
        limit = 0
        bin_width = (1 - limit) / 25
        mask = data > limit
        filtered_data = data[mask]
        percent_show = np.round(np.sum(mask) / len(data), 4) * 100

        bins = np.arange(limit, 1 + bin_width, bin_width)
        counts, bins, patches = ax.hist(filtered_data, bins=bins, edgecolor="black")

        ax.bar_label(patches, labels=[f"{int(c)}" if c > 0 else "" for c in counts], padding=3, fontsize=10)

        ax.set_title(
            f"{prefix}, $R^2$ Scores (Variance Weighted)\n"
            f"{percent_show:.2f}\\% of $R^2$ values are shown, {100 - percent_show:.2f}\\% are smaller than {limit:.1f}"
        )
        ax.set_xlabel("$R^2$ Score")
        ax.set_ylabel("Frequency")
        ax.set_xlim(0, 1.1)

        plt.tight_layout()

    def plot_r2_dist(self, results: Dict[int, str | torch.Tensor]) -> List[float]:
        r2s = []
        for dim, res in results.items():
            res = results[dim]

            tr = res["tr"]
            ys = res["ys"]

            r2 = self.r2_score(tr, ys)
            r2 = r2.cpu().tolist()
            r2s.extend(r2)

            self.plot_r2_distribution(r2, f"Dim: {dim}")

        percent_bigger_90 = np.array(r2s) > 0.9
        percent_bigger_90 = np.mean(percent_bigger_90) * 100
        self.plot_r2_distribution(r2s, f"Overall $R^2$ Distribution ({percent_bigger_90:.2f}\\% $>$ 90 \\%)")

        return r2s

    def get_benchmark_name(self, config: OdeBenchConfig) -> str:
        model_name = self.get_model_id(config)
        out = f"{config.test_type}_{model_name}_n-{config.noise_level}_p-{config.subsample_rate}"

        return out

    def get_model_id(self, config):
        if config.model_type == "odeformer":
            model_name = "odeformer"
        else:
            model_name = str(config.model_path).split("/")[-2]
            model_name = f"{model_name}_ckpt-{config.model_checkpoint}"

        return model_name

    def create_result_folder(self, config: OdeBenchConfig, batch_folder: Path) -> Path:
        model_id = self.get_model_id(config)

        experiment_name = self.get_benchmark_name(config)
        result_folder = batch_folder / model_id / experiment_name
        result_folder.mkdir(parents=True, exist_ok=True)

        return result_folder

    def save_config_info(self, config: OdeBenchConfig, result_folder: Path):
        config_info = {
            "model_type": config.model_type,
            "model_path": str(config.model_path) if config.model_path else None,
            "model_checkpoint": config.model_checkpoint,
            "max_dim": config.max_dim,
            "subsample_rate": config.subsample_rate,
            "noise_level": config.noise_level,
            "test_type": config.test_type,
            "timestamp": datetime.now().isoformat(),
            "device": str(self.device),
        }

        with open(result_folder / "config.json", "w") as f:
            json.dump(config_info, f, indent=4)

    def save_figures_to_folder(self, result_folder: Path, prefix: str = "", subfolder: str = ""):
        save_path = result_folder / subfolder if subfolder else result_folder
        save_path.mkdir(exist_ok=True)

        fig_nums = plt.get_fignums()
        if not fig_nums:
            print("No figures to save.")
            return

        pdf_name = f"{prefix}.pdf" if prefix else "figures.pdf"
        self.save_all_figures_to_pdf(str(save_path / pdf_name))

        print(f"Saved {len(fig_nums)} figures to {save_path}")

    def save_all_figures_to_pdf(self, name: str):
        with PdfPages(name) as pdf:
            fig_nums = plt.get_fignums()
            figs = [plt.figure(n) for n in fig_nums]
            for fig in figs:
                fig.savefig(pdf, format="pdf")
                plt.close(fig)

    def plot_r2_accuracy(
        self,
        all_r2_stats: Dict,
        r2_threshold: float = 0.9,
        test_type: str = "reconstruction",
        model_order: Optional[List[str]] = None,
        model_names: Optional[Dict[str, str]] = {},
    ):
        # I think this is for figures 6.8 and 6.9 in Max's thesis

        # 1. Dynamically discover models, subsample rates, and noise levels from the data
        if model_order is None:
            all_models = sorted([m for m, d in all_r2_stats.items() if test_type in d])
        else:
            all_models = [m for m in model_order if m in all_r2_stats and test_type in all_r2_stats.get(m, {})]

        if not all_models:
            print(f"No data found for test type '{test_type}' and the specified models. Skipping plot.")
            return

        # Discover all unique subsample rates and noise levels present for the selected models
        subsample_rates_found = set()
        noises_found = set()
        for model_id in all_models:
            model_data = all_r2_stats.get(model_id, {}).get(test_type, {})
            subsample_rates_found.update(model_data.keys())
            for rate in model_data:
                noises_found.update(model_data[rate].keys())

        plot_subsample_rates = sorted(subsample_rates_found)
        all_noises = sorted(noises_found)

        if not plot_subsample_rates or not all_noises:
            print("No subsample rates or noise levels found in the data for the selected models. Skipping plot.")
            return

        # 2. Calculate accuracy percentages for all relevant data points
        plot_data = {}
        for model_id in all_models:
            plot_data[model_id] = {}
            for p in plot_subsample_rates:
                plot_data[model_id][p] = {}
                for noise in all_noises:
                    r2s = all_r2_stats.get(model_id, {}).get(test_type, {}).get(p, {}).get(noise, [])
                    plot_data[model_id][p][noise] = np.mean(np.array(r2s) > r2_threshold) * 100 if r2s else np.nan

        # 3. Define Plot Elements
        # Use the user-specified list of potential markers
        potential_markers = ["|", "1", "2", "3", "4", "+", "x", "d"]
        markers = potential_markers[: len(all_noises)]
        colors = plt.cm.magma(np.linspace(0.15, 0.95, len(all_noises)))

        marker_map = dict(zip(all_noises, markers))
        color_map = dict(zip(all_noises, colors))

        # 4. Create a DYNAMIC Plot Layout based on the number of discovered subsample rates
        n_cols = len(plot_subsample_rates)
        n_rows = int(max(len(all_models), 3))
        fig = plt.figure(figsize=(4.5 * n_cols, 1.1 * n_rows))  # Adjust figure width based on number of plots
        fig.set_constrained_layout(True)
        gs_main = gridspec.GridSpec(2, 1, height_ratios=[1.5, 10], hspace=0.01, figure=fig)
        gs_plots = gridspec.GridSpecFromSubplotSpec(1, n_cols, subplot_spec=gs_main[1], wspace=0.1)

        axes = [
            fig.add_subplot(
                gs_plots[0, i],
            )
            for i in range(n_cols)
        ]
        legend_ax = fig.add_subplot(gs_main[0])

        # Share y-axis and hide labels/ticks for all but the first plot
        for ax in axes[1:]:
            ax.sharey(axes[0])
            # Hide the y-axis ticks (the lines) and their labels for a cleaner look
            ax.tick_params(axis="y", left=False, labelleft=False)

        # 5. Populate the Subplots
        y_positions = np.arange(len(all_models))
        for i, p_rate in enumerate(plot_subsample_rates):
            ax = axes[i]

            # Hide the original rectangular spines
            ax.spines[:].set_visible(False)

            # Create and add the new rounded border
            p_fancy = FancyBboxPatch(
                (0, 0),
                1,
                1,  # (x, y) position, width, height
                boxstyle="round,pad=0.0,rounding_size=0.015",
                transform=ax.transAxes,  # Use axes coordinates
                facecolor="none",
                edgecolor="black",
                linewidth=1,
                clip_on=False,  # Allow drawing outside the axes area
                zorder=0.9,  # Draw behind grid lines but on top of background
            )
            ax.add_patch(p_fancy)

            for y_pos, model_id in zip(y_positions, all_models):
                for noise in all_noises:
                    acc = plot_data[model_id][p_rate][noise]
                    if not np.isnan(acc):
                        ax.plot(
                            acc,
                            y_pos,
                            marker=marker_map.get(noise, "o"),
                            color=color_map.get(noise, "black"),
                            markersize=25,
                            markeredgewidth=3,
                            linestyle="None",
                        )

            text_label = rf"$\boldsymbol{{\rho = {p_rate}}}$"

            ax.text(
                0.06,
                0.95,
                text_label,
                transform=ax.transAxes,
                fontsize=20,
                fontweight="bold",  # Makes non-math text bold
                verticalalignment="top",
                horizontalalignment="left",
                bbox={"facecolor": "white", "edgecolor": "white", "boxstyle": "round,pad=0.0"},
            )

            ax.set_xlabel(f"\\% Accuracy ($R^2 > {r2_threshold}$)", fontsize=18, labelpad=12)
            ax.set_xlim(-5, 105)
            ax.set_xticks(np.arange(0, 101, 20))
            ax.grid(True, which="both", linestyle="-", linewidth=0.5)
            ax.tick_params(axis="both", which="major", labelsize=16)

        # Configure y-axis on the leftmost plot
        axes[0].set_yticks(y_positions)

        for k, v in model_names.items():
            if k in all_models:
                all_models[all_models.index(k)] = v

        axes[0].set_yticklabels(all_models, fontsize=19)
        axes[0].invert_yaxis()
        margin = 1  # Adjust this value to increase/decrease margin
        axes[0].set_ylim(-margin, len(all_models) - 1 + margin)

        # 6. Create the Legend
        legend_ax.axis("off")
        legend_handles = [
            Line2D(
                [0],
                [0],
                marker=marker_map.get(n, "o"),
                color=color_map.get(n),
                label=f"$\\sigma={n}$",
                linestyle="None",
                markersize=20,
                markeredgewidth=3,
            )
            for n in all_noises
        ]

        legend_ax.legend(
            handles=legend_handles, ncol=len(all_noises), loc="center", frameon=False, fontsize=18, handletextpad=0.1, columnspacing=1.5
        )

        # fig.suptitle(f"Testmode: {test_type.capitalize()}", fontsize=16)

    def add_r2s(self, data: Dict, config: OdeBenchConfig, r2s: List[float]):
        """
        adds R² scores to a nested dictionary structure for later aggregation and comparison.
        """

        model_id = self.get_model_id(config)
        test_type = config.test_type
        subsample_rate = config.subsample_rate
        noise_level = config.noise_level

        if model_id not in data:
            data[model_id] = {}
        if test_type not in data[model_id]:
            data[model_id][test_type] = {}
        if subsample_rate not in data[model_id][test_type]:
            data[model_id][test_type][subsample_rate] = {}

        data[model_id][test_type][subsample_rate][noise_level] = r2s

    def run(self, configs: List[OdeBenchConfig]):
        all_r2_stats = {}
        batch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_folder = self.results_base_path / f"odebench_run_{batch_timestamp}"
        batch_folder.mkdir(parents=True, exist_ok=True)

        for config in configs:
            benchmark_name = self.get_benchmark_name(config)
            print(f"Running benchmark for config: {benchmark_name}")

            result_folder = self.create_result_folder(config, batch_folder)
            self.current_run_folder = result_folder
            self.save_config_info(config, result_folder)

            print(f"Results will be saved to: {result_folder}")

            ode_per_dim = self.provide_data(config)
            model = self.load_model(config)
            results: Dict[int, Dict[str, torch.Tensor | List[str]]] = self.do_test(ode_per_dim, model, config)
            # results maps the dimension to a dict containing trajectories and predictions etc for all ODEs of that dim.

            with open(result_folder / "results.pkl", "wb") as f:
                pickle.dump(results, f)

            self.save_figures_to_folder(result_folder, "trajectories")
            plt.close("all")

            r2s = self.plot_r2_dist(results)
            self.add_r2s(all_r2_stats, config, r2s)

            self.save_figures_to_folder(result_folder, "r2_distributions")
            plt.close("all")

            with open(result_folder / "r2.json", "w") as f:
                json.dump(r2s, f, indent=4)

            # Clear GPU memory before next config -- not sure if this is right...
            del ode_per_dim, model, results, r2s
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"Completed config: {benchmark_name}, cleared GPU cache")
            print()

        # Create accuracy comparison plots and save to batch folder
        for test_type in ["reconstruction", "generalization"]:
            self.plot_r2_accuracy(all_r2_stats, test_type=test_type, r2_threshold=0.9)
            self.plot_r2_accuracy(all_r2_stats, test_type=test_type, r2_threshold=0.8)

        if plt.get_fignums():
            self.save_figures_to_folder(batch_folder, "accuracy_comparison")
            plt.close("all")

        # Save aggregated R2 statistics
        with open(batch_folder / "all_r2_stats.json", "w") as f:
            json.dump(all_r2_stats, f, indent=4)

        print(f"Batch results saved to: {batch_folder}")
        print(f"Individual run results saved in respective model folders under: {self.results_base_path}")

    def run_for_odeon_model(self, models_path: Path, model_identifier: List[Tuple[str, int]]):
        configs = []
        for model_name, ckpt in model_identifier:
            for test_type in ["reconstruction", "generalization"]:
                for subsample_rate in [0.0, 0.5]:
                    for noise in [0.0, 0.03, 0.05]:
                        config = OdeBenchConfig(
                            model_type="fimodeon",
                            model_path=models_path / model_name / "checkpoints",
                            model_checkpoint=ckpt,
                            max_dim=3,
                            test_type=test_type,
                            noise_level=noise,
                            subsample_rate=subsample_rate,
                        )
                        configs.append(config)

        self.run(configs)

    def run_for_odeformer_model(self):
        configs = []
        for test_type in ["reconstruction", "generalization"]:
            for subsample_rate in [0.0, 0.5]:
                for noise in [0.0, 0.03, 0.05]:
                    config = OdeBenchConfig(
                        model_type="odeformer",
                        max_dim=3,
                        test_type=test_type,
                        noise_level=noise,
                        subsample_rate=subsample_rate,
                    )
                    configs.append(config)

        self.run(configs)


def CLI_entry_point():
    """
    How to use the CLI entry point?

    First activate the venv.

    odebench path/to/model/checkpoints --epoch 200
    where --epoch is optional

    odebench odeformer
    """

    print("sdfjk")

    import argparse

    parser = argparse.ArgumentParser(description="Run ODEBench")
    parser.add_argument("model_path", type=Path, help="Relative path to the model checkpoints inside the results folder.")
    parser.add_argument("--epoch", type=int, default=None, help="Epoch of the model to evaluate.")
    args = parser.parse_args()

    # Initialize OdeBench with path to directory containing strogatz_extended.json
    benchmark_runner = OdeBench(Path("experiments/odebench/data"), device="cuda")
    print("afdj")

    if args.model_path == Path("odeformer"):
        benchmark_runner.run_for_odeformer_model()
    else:
        benchmark_runner.run_for_odeon_model(Path("models"), [(args.model_path, args.epoch)])

    # really it should now also print a summary of the benchmark results...


if __name__ == "__main__":
    """
    To evaluate FIMOdeon and ODEFormer on the ODEBench dataset, we need:
    - strogatz_extended.json in your data directory
    - For FimOdeon: checkpoint directory with model.safetensors (or model-checkpoint.pth) and train-state-checkpoint.pth, plus train_parameters.yaml one level up
    - For OdeFormer: nothing else (downloads automatically)
    """

    print("doing nothing and quitting...")

    quit()

    # # For OdeFormer
    # config = OdeBenchConfig(model_type="odeformer", max_dim=3, test_type="reconstruction", noise_level=0.0, subsample_rate=0.5)
    #
    # benchmark_runner.run([config])
    #
    # quit()
    #
    # p1 = Path("/home/teddev/Downloads/odebench_odeformer-vs-big/odeformer/all_r2_stats.json")
    # p2 = Path("/home/teddev/Downloads/odebench_odeformer-vs-big/big/all_r2_stats.json")
    #
    # plt.rcParams["font.family"] = "TeX Gyre Pagella"
    # plt.rcParams.update(
    #     {
    #         "text.usetex": True,
    #         "text.latex.preamble": r"\usepackage{amsmath}",
    #     }
    # )
    #
    # benchmark_runner = OdeBench(Path("scripts/fimodeon_scripts/exploration_new"))
    #
    # with open(p1, "r") as f:
    #     of_all_r2_stats = json.load(f)
    # with open(p2, "r") as f:
    #     big_all_r2_stats = json.load(f)
    #
    # all_r2_stats = {**of_all_r2_stats, **big_all_r2_stats}
    #
    # names = {"odeformer": "ODEFormer", "big_model_l1_600k_examples_07-29-1659_ckpt-200": "FIM"}
    #
    # benchmark_runner.plot_r2_accuracy(all_r2_stats, r2_threshold=0.8, test_type="generalization", model_names=names)
    #
    # # plt.show()
    # plt.savefig("odebench_generalization_08_threshold.pdf")
    #
    # quit()
    #
    # benchmark_runner = OdeBench(
    #     Path(
    #         # "/lustre/mlnvme/data/s78mmaue_hpc-demo2/fim_training/FIM2/scripts/fimodeon_scripts/exploration_new"
    #         "/home/teddev/PycharmProjects/FIM/scripts/fimodeon_scripts/exploration_new"
    #     )
    # )
    #
    # benchmark_runner.run_for_odeformer_model()
    # quit()
    #
    # models = [("new_data_hub_5kpoints_05-17-1706", 55)]
    #
    # benchmark_runner.run_for_odeon_model(
    #     Path(
    #         # f"/lustre/mlnvme/data/s78mmaue_hpc-demo2/fim_training/FIM2/results"
    #         "/home/teddev/PycharmProjects/FIM/scripts/results"
    #     ),
    #     models,
    # )
    #
    # # config = OdeBenchConfig(
    # #     model_type="odeformer",
    # #     max_dim=3,
    # #     test_type="reconstruction",
    # # )
    # #
    # # benchmark_runner.run([config])
