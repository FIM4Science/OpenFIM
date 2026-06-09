from pathlib import Path
import pickle
import torch
import numpy as np
from scipy.integrate import solve_ivp
from utils.h5 import get_shape_of_h5, get_dtype_of_h5, get_type_of_h5, parse_h5, print_h5, get_ndarray_from_h5
from utils.helpers import load_odeon_model_from_checkpoint, predict_vector_field, predict_and_integrate_ode, plot_vector_field_and_trajectories
from ODEs import VDP_ode, FHN_ode
import statistics
from utils.eval_models import PredictionModel, OdeFormerEval
from utils.data_models import trajectory, trajectory_list_from_h5_files
from utils.plot import plot_3d_paths
from data_gen_mocap import Data, MocapDataset, Normalize
from sklearn.decomposition import PCA
from other.data_gen_vdp_fhn import ExperimentData
from tqdm import tqdm
from itertools import combinations
import random
import sys
import traceback
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# Single experiments:

def vdp_task_1(model: PredictionModel, data_path: Path, plot=False, plot_type: str = "2D"):
    context_xs = get_ndarray_from_h5(data_path / "obs_values.h5")       # always (1, 1, 50, 2)
    context_ts = get_ndarray_from_h5(data_path / "obs_times.h5")        # always (1, 1, 50, 1)
    context_traj = trajectory(context_xs, context_ts)

    # Predict vector field and integrate
    y0 = np.array([-1.5, 2.5])
    t_eval: np.ndarray = np.linspace(7,14,100)
    pred_traj_second_half: trajectory = predict_and_integrate_ode(model, [context_traj], y0, t_eval)

    # next need to compute the MSE over 7..14 -- but MSE distance from what? From the ground-truth trajectory, NOT the noisy observations
    # Integrate from t=0 with y0 at t=0, then extract values at t_eval
    time_span = (0.0, 14.0)
    true_xs_second_half = solve_ivp(VDP_ode, time_span, y0, t_eval=t_eval, method='RK45').y.T   # (100,2)

    if plot:
        true_xs_first_half = solve_ivp(VDP_ode, (0.,7.), y0, t_eval=np.linspace(0,7,100), method='RK45').y.T    # (100,2)
        true_traj_first_half = trajectory(true_xs_first_half, np.linspace(0,7,100))
        true_traj_second_half = trajectory(true_xs_second_half, t_eval)

        pred_traj_first_half: trajectory = predict_and_integrate_ode(model, [context_traj], y0, np.linspace(0,7,100))

        if isinstance(plot, matplotlib.axes.Axes):
            if plot_type == "2D":
                plot_vector_field_and_trajectories(model, context_trajs_sparse=[context_traj], predicted_trajs = [pred_traj_first_half, pred_traj_second_half], true_trajs=[true_traj_first_half, true_traj_second_half], ax=plot)
            elif plot_type == "2D_streamplot":
                plot_vector_field_and_trajectories(model, context_trajs_sparse=[context_traj], predicted_trajs = [pred_traj_first_half, pred_traj_second_half], true_trajs=[true_traj_first_half, true_traj_second_half], ax=plot, plot_type="2D_streamplot", show_legend=False, draw_start_and_end_points=False)
            elif plot_type == "1D":
                raise NotImplementedError("1D plot type not implemented yet")
                plot_vector_field_and_trajectories(model, context_trajs_sparse=[context_traj], predicted_trajs = [pred_traj_first_half, pred_traj_second_half], true_trajs=[true_traj_first_half, true_traj_second_half], ax=plot, plot_type="1D")
        else:
            plot_vector_field_and_trajectories(model, context_trajs_sparse=[context_traj], predicted_trajs = [pred_traj_first_half, pred_traj_second_half], true_trajs=[true_traj_first_half, true_traj_second_half])
    
    MSE = ((pred_traj_second_half.xs - true_xs_second_half) ** 2).mean()   # You may think this underestimates by factor 2, but that's how the GP-ODE repo does it...
    return MSE

def vdp_task_2(model: PredictionModel, data_path: Path, plot=False, plot_type="2D"):
    context_xs = get_ndarray_from_h5(data_path / "obs_values.h5")       # always (1, 1, 50, 2)
    context_ts = get_ndarray_from_h5(data_path / "obs_times.h5")        # always (1, 1, 50, 1)
    context_traj = trajectory(context_xs, context_ts)

    # Predict vector field and integrate
    y0 = np.array([-1.5, 2.5])
    t_eval: np.ndarray = np.linspace(7,14,100)
    pred_traj_second_half: trajectory = predict_and_integrate_ode(model, [context_traj], y0, t_eval) # np.ndarray of shape (100,2)

    #plot_vector_field_and_trajectories(model, context_trajs=[context_traj], context_time_grids=[context_times], predicted_trajs=[predicted_trajectory])

    # next need to compute the MSE over 7..14 -- but MSE distance from what? From the ground-truth trajectory, NOT the noisy observations
    # Integrate from t=0 with y0 at t=0, then extract values at t_eval
    time_span = (0.0, 14.0)
    true_xs_second_half = solve_ivp(VDP_ode, time_span, y0, t_eval=t_eval, method='RK45').y.T # numpy array of shape (100,2)

    if plot:
        true_xs_first_half = solve_ivp(VDP_ode, (0.,7.), y0, t_eval=np.linspace(0,7,100), method='RK45').y.T    # (100,2)
        true_traj_first_half = trajectory(true_xs_first_half, np.linspace(0,7,100))
        true_traj_second_half = trajectory(true_xs_second_half, t_eval)

        pred_traj_first_half: trajectory = predict_and_integrate_ode(model, [context_traj], y0, np.linspace(0,7,100))

        if isinstance(plot, matplotlib.axes.Axes):
            if plot_type == "2D":
                plot_vector_field_and_trajectories(model, context_trajs_sparse=[context_traj], predicted_trajs = [pred_traj_first_half, pred_traj_second_half], true_trajs=[true_traj_first_half, true_traj_second_half], ax=plot)
            elif plot_type == "2D_streamplot":
                plot_vector_field_and_trajectories(model, context_trajs_sparse=[context_traj], predicted_trajs = [pred_traj_first_half, pred_traj_second_half], true_trajs=[true_traj_first_half, true_traj_second_half], ax=plot, plot_type="2D_streamplot", show_legend=False, draw_start_and_end_points=False)
            elif plot_type == "1D":
                plot_vector_field_and_trajectories(model, context_trajs_sparse=[context_traj], predicted_trajs = [pred_traj_first_half, pred_traj_second_half], true_trajs=[true_traj_first_half, true_traj_second_half], ax=plot, plot_type="1D")
        else:
            plot_vector_field_and_trajectories(model, context_trajs_sparse=[context_traj], predicted_trajs = [pred_traj_first_half, pred_traj_second_half], true_trajs=[true_traj_first_half, true_traj_second_half])
    
    MSE = ((pred_traj_second_half.xs - true_xs_second_half)**2).mean()   # You may think this underestimates by factor 2, but that's how the GP-ODE repo does it...
    #print("MSE Vanderpol Task 2:", MSE)
    return MSE

def fhn_task(model: PredictionModel, data_path: Path, fine_MSE=False, plot=False, plot_type="2D"):

    # Load data, obs_mask.h5 may be ignored; it is of shape (1,1,19,1) and all Trues
    context_xs = get_ndarray_from_h5(data_path / "obs_values.h5")   # (1, 1, 19, 2)
    context_ts = get_ndarray_from_h5(data_path / "obs_times.h5")    # (1, 1, 19, 1)
    context_traj = trajectory(context_xs, context_ts)               # length 19, dimension 2

    # Predict vector field and integrate
    y0 = np.array([-1., -1.])       # initial condition
    t_eval: np.ndarray = np.linspace(0,5,1000)
    pred_traj: np.ndarray = predict_and_integrate_ode(model, [context_traj],  y0, t_eval) # np.ndarray of shape (1000,2)

    # next need to compute the MSE on 4th quadrant -- but MSE distance from what? From the ground-truth trajectory, NOT the noisy observations
    time_span = (0.0, 5.0)
    sol = solve_ivp(FHN_ode, time_span, y0, t_eval=t_eval, method='RK45')
    true_traj = trajectory(sol.y.T, t_eval) # numpy array of shape (1000,2)

    if plot:
        if isinstance(plot, matplotlib.axes.Axes):
            if plot_type == "2D":
                plot_vector_field_and_trajectories(model, context_trajs_sparse=[context_traj], predicted_trajs=[pred_traj], true_trajs=[true_traj], ax=plot, plot_type="2D")
            elif plot_type == "2D_streamplot":
                # print(context_traj.xs)
                # print(pred_traj.xs)
                # print(true_traj.xs)
                plot_vector_field_and_trajectories(model, context_trajs_sparse=[context_traj], predicted_trajs=[pred_traj], true_trajs=[true_traj], ax=plot, plot_type="2D_streamplot", show_legend=False, draw_start_and_end_points=False)
            elif plot_type == "1D":
                plot_vector_field_and_trajectories(model, context_trajs_sparse=[context_traj], predicted_trajs=[pred_traj], true_trajs=[true_traj], ax=plot, plot_type="1D")
        else:
            plot_vector_field_and_trajectories(model, context_trajs_sparse=[context_traj], predicted_trajs=[pred_traj], true_trajs=[true_traj])

    if fine_MSE:
        quadrant_mask = (true_traj.xs[:,0]>0) & (true_traj.xs[:,1]<0)       # (1000,)
        # print("abacca")
        # print(quadrant_mask.shape)
        # print(quadrant_mask.sum())
        MSE = (((pred_traj.xs - true_traj.xs) ** 2) * np.column_stack((quadrant_mask,quadrant_mask))).mean() * 1000 / quadrant_mask.sum()   # It's unclear what this factor should be...
        return MSE
    else:
        pkl_path = data_path / "experiment_data.pkl"
        if pkl_path.exists():
            with open(data_path / "experiment_data.pkl", 'rb') as f:
                experiment_data: ExperimentData = pickle.load(f) # contains test_trajectory (25,2), test_time_grid (25,), loss_mask (25,2)
                # experiment_data.loss_mask[:,0] and experiment_data.loss_mask[:,1] are the same
                mask = experiment_data.loss_mask
                # mask is True exactly 6 times (2 disjoint segments), False 19 times

                #test_xs = experiment_data.test_trajectory
                test_ts = experiment_data.test_time_grid#[mask[:,0]]

                pred_xs = predict_and_integrate_ode(model, [context_traj], y0, test_ts).xs          # (25,2)
                true_xs = solve_ivp(FHN_ode, time_span, y0, t_eval=test_ts, method='RK45').y.T   # (25,2)

                #print(((pred_xs - true_xs) ** 2) * mask)
                MSE = (((pred_xs - true_xs) ** 2) * experiment_data.loss_mask).sum() / 12.0      # It's unclear what this factor should be... I guess judging by the implementation for VDP, I should divide by 12 here
        else:
            test_ts = get_ndarray_from_h5(data_path / "test_times.h5")  # (1,1,6,1)
            test_xs = get_ndarray_from_h5(data_path / "test_values.h5") # (1,1,6,2)

            #pred_xs = predict_and_integrate_ode(model, [context_traj], y0, test_ts.reshape(6,)).xs     # (6,2)
            pred_xs = predict_and_integrate_ode(model, [context_traj], y0, test_ts.reshape(5,)).xs     # (6,2)
            #pred_xs.reshape(1,1,6,2)
            pred_xs.reshape(1,1,5,2)

            #true_xs = solve_ivp(FHN_ode, time_span, y0, t_eval=test_ts.reshape(6,)).y.T   # (6,2)
            true_xs = solve_ivp(FHN_ode, time_span, y0, t_eval=test_ts.reshape(5,)).y.T   # (6,2)

            # Use true_xs or test_xs for MSE?
            MSE = ((pred_xs - true_xs) ** 2).sum() / 10. #/ 12.0       # It's unclear what this factor should be... I guess judging by the implementation for VDP, I should divide by 12 here

        return MSE

def mocap_task(model: PredictionModel, data_path: Path, plot=False, dim_left=3, left_only=False, sublist: tuple = None):
    ctx_trajs_5d: list[trajectory] = trajectory_list_from_h5_files(path_to_xs=data_path/"obs_values.h5", path_to_ts=data_path/"obs_times.h5")

    def split_5d(traj_list_5d: list[trajectory]):
        # 3+2 or 2+3 (depending on whether dim_left is 3 or 2) split of dimensions is necessary
        traj_list_left = []
        traj_list_right = []
        for traj_5d in traj_list_5d:
            ts = traj_5d.ts

            xs_5d = traj_5d.xs
            #xs_left = xs_5d[:, [0,3,4]]
            #xs_right = xs_5d[:, [1,2]]
            xs_left = xs_5d[:, :dim_left]
            xs_right = xs_5d[:, dim_left:]

            traj_left = trajectory(xs=xs_left, ts=ts)
            traj_list_left.append(traj_left)

            traj_right = trajectory(xs=xs_right, ts=ts)
            traj_list_right.append(traj_right)

        return (traj_list_left, traj_list_right)

    ctx_trajs_left, ctx_trajs_right = split_5d(ctx_trajs_5d)        # trajectory length: 50 (subject 35 short)
    
    # Load testing trajectories -- we do not use these for inference, but for initial conditions and to compare
    with open(data_path / "mocap_dataset.pkl", 'rb') as f:
        dataset: MocapDataset = pickle.load(f)
    
    test_trajs_5d = []
    # dataset.tst: Data
    for i in range(dataset.tst.ys.shape[0]):
        # dataset.tst.ys (2,120,5)
        # dataset.tst.ts (120,)
        test_trajs_5d.append(trajectory(dataset.tst.ys[i,:,:], dataset.tst.ts))
        
    test_trajs_left, test_trajs_right = split_5d(test_trajs_5d)     # trajectory length: 300 (subject 35)

    # evaluation time grid for predicted trajectories in left and right
    pred_ts = test_trajs_5d[0].ts

    # left prediction & ODE integration:
    if sublist is None:
        pred_trajs_left = [
            predict_and_integrate_ode(model, ctx_trajs_left, traj_left.xs[0,:], pred_ts) for traj_left in test_trajs_left
        ]
    else:
        pred_trajs_left = [
            predict_and_integrate_ode(model, [ctx_trajs_left[i] for i in sublist],
                            traj_left.xs[0,:], pred_ts) for traj_left in test_trajs_left
        ]
    
    if plot:
        if dim_left == 3: plot_3d_paths(pred_list=pred_trajs_left[:], ctx_list=ctx_trajs_left, obs_list=test_trajs_left)
        if dim_left == 2: plot_vector_field_and_trajectories(model, context_trajs=ctx_trajs_left, predicted_trajs=pred_trajs_left)

    if left_only:
        # right prediction is just zeroes
        pred_trajs_right = [trajectory(np.zeros((pred_trajs_left[0].length, 5-dim_left)), pred_ts) for _ in pred_trajs_left]
    else:
        # right prediction:
        pred_trajs_right = [
            predict_and_integrate_ode(model, ctx_trajs_right, traj_right.xs[0,:], pred_ts) for traj_right in test_trajs_right
        ]

    # Combine left and right predictions into 5D trajectories
    pred_trajs_5d = [trajectory(xs=np.column_stack([pred_traj_left.xs, pred_traj_right.xs]), ts=pred_ts)
                            for pred_traj_left, pred_traj_right in zip(pred_trajs_left, pred_trajs_right)]
    
    arr_pred_5d = np.stack([pred_traj.xs for pred_traj in pred_trajs_5d], axis=0)   # (2,120,5)
    arr_test_5d = np.stack([test_traj.xs for test_traj in test_trajs_5d], axis=0)   # (2,120,5)

    # Invert PCA's normalization:
    arr_pred_5d = dataset.pca_normalize.inverse(arr_pred_5d)    # (2,120,5)
    arr_test_5d = dataset.pca_normalize.inverse(arr_test_5d)    # (2,120,5)

    # Invert PCA:
    arr_pred_50d = dataset.pca.inverse_transform(arr_pred_5d)   # (2,120,50)
    arr_test_50d = dataset.pca.inverse_transform(arr_test_5d)   # (2,120,50)

    MSE = ((arr_pred_50d - arr_test_50d)**2).mean()
    return(MSE)

def mocap_task_new(model: PredictionModel, data_path: Path, plot=False, dim_left=3, left_only=False, sublist: tuple = None):
    ctx_trajs_5d: list[trajectory] = trajectory_list_from_h5_files(path_to_xs=data_path/"obs_values.h5", path_to_ts=data_path/"obs_times.h5")
    
    def split_5d(traj_list_5d: list[trajectory]):
        # [0,1,2] and [2,3,4]
        traj_list_left = []
        traj_list_right = []
        for traj_5d in traj_list_5d:
            ts = traj_5d.ts

            xs_5d = traj_5d.xs
            xs_left = xs_5d[:, [0,1,2]]
            xs_right = xs_5d[:, [1,3,4]]

            traj_left = trajectory(xs=xs_left, ts=ts)
            traj_list_left.append(traj_left)

            traj_right = trajectory(xs=xs_right, ts=ts)
            traj_list_right.append(traj_right)

        return (traj_list_left, traj_list_right)

    ctx_trajs_left, ctx_trajs_right = split_5d(ctx_trajs_5d)        # trajectory length: 50 (subject 35 short)
    
    # Load testing trajectories -- we do not use these for inference, but for initial conditions and to compare
    with open(data_path / "mocap_dataset.pkl", 'rb') as f:
        dataset: MocapDataset = pickle.load(f)
    
    test_trajs_5d = []
    # dataset.tst: Data
    for i in range(dataset.tst.ys.shape[0]):
        # dataset.tst.ys (2,120,5)
        # dataset.tst.ts (120,)
        test_trajs_5d.append(trajectory(dataset.tst.ys[i,:,:], dataset.tst.ts))
        
    test_trajs_left, test_trajs_right = split_5d(test_trajs_5d)     # trajectory length: 300 (subject 35)

    # evaluation time grid for predicted trajectories in left and right
    pred_ts = test_trajs_5d[0].ts

    # left prediction & ODE integration:
    if sublist is None:
        pred_trajs_left = [
            predict_and_integrate_ode(model, ctx_trajs_left, traj_left.xs[0,:], pred_ts) for traj_left in test_trajs_left
        ]
    else:
        pred_trajs_left = [
            predict_and_integrate_ode(model, [ctx_trajs_left[i] for i in sublist],
                            traj_left.xs[0,:], pred_ts) for traj_left in test_trajs_left
        ]
    
    if plot:
        if isinstance(plot, matplotlib.axes.Axes):
            plot_3d_paths(pred_list=pred_trajs_left[:], ctx_list=ctx_trajs_left, obs_list=test_trajs_left, ax=plot)
        else:
            plot_3d_paths(pred_list=pred_trajs_left[:], ctx_list=ctx_trajs_left, obs_list=test_trajs_left)

    if left_only:
        # right prediction is just zeroes
        pred_trajs_right = [trajectory(np.zeros((pred_trajs_left[0].length, 2)), pred_ts) for _ in pred_trajs_left]
    else:
        # right prediction:
        pred_trajs_right = [
            predict_and_integrate_ode(model, ctx_trajs_right, traj_right.xs[0,:], pred_ts) for traj_right in test_trajs_right
        ]

    # Combine left and right predictions into 5D trajectories
    if left_only:
        pred_trajs_5d = [trajectory(xs=np.column_stack([left.xs, right.xs]), ts=pred_ts) for left, right in zip(pred_trajs_left, pred_trajs_right)]
    else:
        pred_trajs_5d = [trajectory(xs=np.column_stack([left.xs, right.xs[:,1:]]), ts=pred_ts) for left, right in zip(pred_trajs_left, pred_trajs_right)]
    
    arr_pred_5d = np.stack([pred_traj.xs for pred_traj in pred_trajs_5d], axis=0)   # (2,120,5)
    arr_test_5d = np.stack([test_traj.xs for test_traj in test_trajs_5d], axis=0)   # (2,120,5)

    # Invert PCA's normalization:
    arr_pred_5d = dataset.pca_normalize.inverse(arr_pred_5d)    # (2,120,5)
    arr_test_5d = dataset.pca_normalize.inverse(arr_test_5d)    # (2,120,5)

    # Invert PCA:
    arr_pred_50d = dataset.pca.inverse_transform(arr_pred_5d)   # (2,120,50)
    arr_test_50d = dataset.pca.inverse_transform(arr_test_5d)   # (2,120,50)

    MSE = ((arr_pred_50d - arr_test_50d)**2).mean()
    return(MSE)


def plot_vdp(base_model: str, finetuned_model: str):
    pass


def plot_fhn(ax: plt.Axes):
    """Plot true FHN vector field (streamplot), integral curve, and noisy observations."""
    data_path = Path("experiments/fhn/data_gpode")
    
    # Load noisy observations
    context_xs = get_ndarray_from_h5(data_path / "obs_values.h5")   # (1, 1, T, 2)
    context_ts = get_ndarray_from_h5(data_path / "obs_times.h5")    # (1, 1, T, 1)
    context_traj = trajectory(context_xs, context_ts)
    obs_points = context_traj.xs.squeeze()  # (T, 2)
    
    # Determine plot range from observations
    x_min, x_max = obs_points[:, 0].min() - 0.5, obs_points[:, 0].max() + 0.5
    y_min, y_max = obs_points[:, 1].min() - 0.5, obs_points[:, 1].max() + 0.5
    
    # Create grid for vector field
    grid_resolution = 30
    x = np.linspace(x_min, x_max, grid_resolution)
    y = np.linspace(y_min, y_max, grid_resolution)
    X, Y = np.meshgrid(x, y)
    grid_points = np.stack([X.ravel(), Y.ravel()], axis=1)  # (N, 2)
    
    # Compute true FHN vector field at grid points
    vf = np.array([FHN_ode(0, point) for point in grid_points])  # (N, 2)
    U = vf[:, 0].reshape(X.shape)
    V = vf[:, 1].reshape(Y.shape)
    
    # Plot streamplot of vector field
    ax.streamplot(X, Y, U, V, color="#dddddd", density=0.6, linewidth=1.5, arrowsize=1.5)
    
    # Compute and plot true integral curve
    y0 = np.array([-1., -1.])
    t_eval = np.linspace(0, 5, 1000)
    sol = solve_ivp(FHN_ode, (0.0, 5.0), y0, t_eval=t_eval, method='RK45')
    true_traj_points = sol.y.T  # (1000, 2)
    ax.plot(true_traj_points[:, 0], true_traj_points[:, 1], 
            "b-", label="True integral curve", linewidth=2, alpha=0.8)
    
    # Plot noisy observations
    ax.scatter(obs_points[:, 0], obs_points[:, 1], 
               color="red", s=40, marker="x", label="Noisy observations", 
               alpha=0.9, zorder=5, edgecolors="black", linewidths=2)
    
    ax.set_xlabel("$x_1$", fontsize=14)
    ax.set_ylabel("$x_2$", fontsize=14)
    ax.set_title("FHN: True vector field, integral curve, and observations", fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")


def plot_fhn_comparison_for_paper(base_model: PredictionModel, finetuned_model: PredictionModel):
    fig, ax = plt.subplots(ncols=3, figsize=(30, 10))
    plot_fhn(ax[0])
    fhn_task(base_model, Path("experiments/fhn/data_gpode"), fine_MSE=False, plot=ax[1], plot_type="2D_streamplot")
    fhn_task(finetuned_model, Path("experiments/fhn/data_gpode"), fine_MSE=False, plot=ax[2], plot_type="2D_streamplot")
    for a in ax:
        a.grid(False)
    fig.tight_layout()
    plt.savefig('experiments/fhn/fhn_comparison.pdf', bbox_inches="tight")
    plt.close()


def sin(ax):
    x = np.linspace(0, 2*np.pi, 400)
    ax.plot(x, np.sin(x), label=r'$\sin(x)$')

def save_summary_pdf(model: PredictionModel, filename: str = "experiments/summary.pdf"):
    with PdfPages(filename) as pdf:
        fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(16, 20), subplot_kw={'projection': '3d'})
        mocap09short = mocap_task_new(model, Path("experiments/mocap/mocap09short/data/train/5d"), left_only=True, plot=axs[0,0])
        axs[0,0].set_title(f"Mocap 09 Short: {mocap09short:.3f}")
        mocap09long = mocap_task_new(model, Path("experiments/mocap/mocap09long/data/train/5d"), left_only=True, plot=axs[0,1])
        axs[0,1].set_title(f"Mocap 09 Long: {mocap09long:.3f}")

        mocap35short = mocap_task_new(model, Path("experiments/mocap/mocap35short/data/train/5d"), left_only=True, plot=axs[1,0])
        axs[1,0].set_title(f"Mocap 35 Short: {mocap35short:.3f}")
        mocap35long = mocap_task_new(model, Path("experiments/mocap/mocap35long/data/train/5d"), left_only=True, plot=axs[1,1])
        axs[1,1].set_title(f"Mocap 35 Long: {mocap35long:.3f}")
        
        mocap39short = mocap_task_new(model, Path("experiments/mocap/mocap39short/data/train/5d"), left_only=True, plot=axs[2,0])
        axs[2,0].set_title(f"Mocap 39 Short: {mocap39short:.3f}")
        mocap39long = mocap_task_new(model, Path("experiments/mocap/mocap39long/data/train/5d"), left_only=True, plot=axs[2,1])
        axs[2,1].set_title(f"Mocap 39 Long: {mocap39long:.3f}")

        fig.tight_layout() 
        pdf.savefig(fig)
        plt.close(fig)
        
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 8))
        vdp1 = vdp_task_1(model, Path("experiments/vdp1/data_gpode"), plot=ax1, plot_type="2D_streamplot")
        ax1.set_title(f"VDP 1: {vdp1:.3f}")
        
        vdp2 = vdp_task_2(model, Path("experiments/vdp2/data_gpode"), plot=ax2, plot_type="2D")
        ax2.set_title(f"VDP 2: {vdp2:.3f}")
        
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


        fig, ax1 = plt.subplots(figsize=(8, 8))
        fhn = fhn_task(model, Path("experiments/fhn/data_gpode"), fine_MSE=False, plot=ax1, plot_type="2D_streamplot")
        ax1.set_title(f"FHN: {fhn:.3f}")
        pdf.savefig(fig)
        plt.close(fig)
        



if __name__ == "__main__":
    #base_model = load_odeon_model_from_checkpoint(Path("models/base_model/checkpoints"), epoch=None)
    #finetuned_model = load_odeon_model_from_checkpoint(Path("models/fhn/gridsearch_01-28-2120/checkpoints"), epoch=None)

    model = OdeFormerEval()
    #print( fhn_task(model, Path("experiments/fhn/data_gpode"), fine_MSE=False, plot=True, plot_type="2D") )
    print( fhn_task(model, Path("experiments/fhn/data_gpode"), fine_MSE=True, plot=True, plot_type="2D") )
    print(vdp_task_1(model, Path("experiments/vdp1/data_gpode"), plot=True))
    print(vdp_task_2(model, Path("experiments/vdp2/data_gpode"), plot=True))

    model = load_odeon_model_from_checkpoint(Path("models/new_pretraining_1_continued/checkpoints"), epoch=None)
    save_summary_pdf(model, filename="experiments/summary_pretraining.pdf")
    del model
    quit()

    #model = load_odeon_model_from_checkpoint(Path("models/posttraining_3/checkpoints"), epoch=None)
    #save_summary_pdf(model, filename="experiments/summary_posttraining.pdf")
    #del model

    model = load_odeon_model_from_checkpoint(Path("models/posttraining_3/checkpoints"), epoch=None)
    save_summary_pdf(model, filename="experiments/summary_posttraining.pdf")
    del model
    quit()

    print( f"{mocap_task_new(model, Path("experiments/mocap/mocap09short/data/train/5d"), plot=False):.2f}" )
    print( f"{mocap_task_new(model, Path("experiments/mocap/mocap09long/data/train/5d"), plot=False):.2f}" )
    print( f"{mocap_task_new(model, Path("experiments/mocap/mocap35short/data/train/5d"), plot=False):.2f}" )
    print( f"{mocap_task_new(model, Path("experiments/mocap/mocap35long/data/train/5d"), plot=False):.2f}" )
    print( f"{mocap_task_new(model, Path("experiments/mocap/mocap39short/data/train/5d"), plot=False):.2f}" )
    print( f"{mocap_task_new(model, Path("experiments/mocap/mocap39long/data/train/5d"), plot=False):.2f}" )
    
    print("="*100)
    print(vdp_task_1(model, Path("experiments/vdp1/data_gpode"), plot=True))
    print(vdp_task_2(model, Path("experiments/vdp2/data_gpode"), plot=False))
    print(fhn_task(model, Path("experiments/fhn/data_gpode"), fine_MSE=False, plot=False))
    
    #plots_for_paper()

    quit()
    #model = OdeFormerEval()    # Note that ODEFormer doesn't work for the MoCap experiments as it only accepts one trajectory! Tasks 1-3 are fine though.
    model = load_odeon_model_from_checkpoint(Path("models/base_model/checkpoints"))
    print( f"{mocap_task_new(model, Path("experiments/mocap/mocap09short/data/train/5d"), plot=False):.2f}" )
    print( f"{mocap_task_new(model, Path("experiments/mocap/mocap09long/data/train/5d"), plot=False):.2f}" )
    print( f"{mocap_task_new(model, Path("experiments/mocap/mocap35short/data/train/5d"), plot=False):.2f}" )
    print( f"{mocap_task_new(model, Path("experiments/mocap/mocap35long/data/train/5d"), plot=False):.2f}" )
    print( f"{mocap_task_new(model, Path("experiments/mocap/mocap39short/data/train/5d"), plot=False):.2f}" )
    print( f"{mocap_task_new(model, Path("experiments/mocap/mocap39long/data/train/5d"), plot=False):.2f}" )
    

    #print(vdp_task_1(model, Path("experiments/vdp1/data_gpode"), plot=True))
    #print(vdp_task_2(model, Path("experiments/vdp2/data_gpode"), plot=True))
    #print(fhn_task(model, Path("experiments/fhn/data_gpode"), fine_MSE=False, plot=True))
    #print(fhn_task(model, Path("experiments/fhn/data_gpode"), fine_MSE=True, plot=True))

    quit()
    
    #model = load_odeon_model_from_checkpoint(Path("models/vdp1"))
    #model = load_odeon_model_from_checkpoint(Path("models/vdp2")
    #model = load_odeon_model_from_checkpoint(Path("models/fhn")
    
    """
    data_path = Path("experiments/vdp1/data")
    print(vdp_task_1(model, model, data_path / "0", plot=False))
    #N_times_VDP_task_1(model, data_path, N_datasets=100, plot=False, verbose=False)

    data_path = Path("experiments/vdp1/data")
    print(vdp_task_2(model, data_path, plot=False))

    data_path = Path("experiments/fhn/data")
    print(fhn_task(model, data_path, fine_MSE=False, plot=False))
    print(fhn_task(model, data_path, fine_MSE=True, plot=False))
    """


    data_path = Path("experiments/mocap/")
    """
    for task_name in ["mocap09long", "mocap09short", "mocap35long", "mocap35short", "mocap39long", "mocap39short"]:
        print()
        print(task_name)
        for d in (Path("models/") / task_name).iterdir():
            if d.is_dir():
                for e in d.iterdir():
                    model = load_odeon_model_from_checkpoint(e / "checkpoints", epoch=None)
                    print( mocap_task(model, Path("experiments/mocap/") / task_name / "data" / "5d", plot=True, dim_left=3, left_only=True, sublist=None) )
    """

    model = load_odeon_model_from_checkpoint("models/vdp2/vdp2_losssteps=49_hmax=0.025_loss=mse/vdp2_losssteps=49_hmax=0.025_loss=mse_01-16-1128/checkpoints", epoch=None)
    print(vdp_task_2(model, Path("experiments/vdp2/data") / "0", plot=True))
    for i in range(100):
        model = load_odeon_model_from_checkpoint("models/vdp2/vdp2_losssteps=49_hmax=0.025_loss=mse/vdp2_losssteps=49_hmax=0.025_loss=mse_01-16-1128/checkpoints", epoch=201+i)
        print(201+i, vdp_task_2(model, Path("experiments/vdp2/data") / "0", plot=False))
        pass
    quit()

    model = load_odeon_model_from_checkpoint("models/vdp1/vdp1_losssteps=49_ninter=2_loss=mse/vdp1_losssteps=49_ninter=2_loss=mse_01-15-2324/checkpoints", epoch=None)
    print(vdp_task_1(model, Path("experiments/vdp1/data") / "0", plot=True))
    """
    for i in range(100):
        model = load_odeon_model_from_checkpoint("models/fhn/fhn_losssteps=18_ninter=10_loss=mse/fhn_losssteps=18_ninter=10_loss=mse_01-16-0035/checkpoints", epoch=201+i)
        print(201+i, vdp_task_1(model, Path("experiments/vdp1/data") / "0", plot=False))
        # Best epoch: 232 with 0.025422130243601355
        pass
    """

    # 1 step:
    model = load_odeon_model_from_checkpoint("models/fhn/fhn_losssteps=3_hmax=0.001_loss=mse/fhn_losssteps=3_hmax=0.001_loss=mse_01-16-1047/checkpoints", epoch=None)
    print(fhn_task(model, Path("experiments/fhn/data") / "0", fine_MSE=False, plot=True))
    for i in range(100):
        model = load_odeon_model_from_checkpoint("models/fhn/fhn_losssteps=3_hmax=0.001_loss=mse/fhn_losssteps=3_hmax=0.001_loss=mse_01-16-1047/checkpoints", epoch=201+i)
        print(201+i, fhn_task(model, Path("experiments/fhn/data") / "0", fine_MSE=False, plot=False))
        pass


    quit()
    model = load_odeon_model_from_checkpoint("models/vdp1/vdp1_losssteps=49_ninter=2_loss=mse/vdp1_losssteps=49_ninter=2_loss=mse_01-15-2324/checkpoints", epoch=None)
    #N_times_VDP_task_1(model, data_path=Path("experiments/vdp1/data"), N_datasets=50, plot=False, verbose=False)
    N_times_VDP_task_2(model, data_path=Path("experiments/vdp2/data"), N_datasets=50, plot=False, verbose=False)
    #N_times_FHN_task(model, data_path=Path("experiments/fhn/data"), N_datasets=20, fine_MSE=False, plot=False, verbose=False)
    #print(vdp_task_1(model, Path("experiments/vdp1/data") / "2", plot=True))

    quit()
    # config_noise_0.yaml
    if True:
        first_model = "models/vdp1/vdp1_losssteps=15_ninter=16_loss=mse/vdp1_losssteps=15_ninter=16_loss=mse_01-15-1949/checkpoints"
        second_model = "models/vdp1/vdp1_losssteps=15_ninter=16_loss=mse/vdp1_losssteps=15_ninter=16_loss=mse_01-15-2127/checkpoints"
        model_l1 = "models/vdp1/vdp1_losssteps=15_ninter=16_loss=l1/vdp1_losssteps=15_ninter=16_loss=l1_01-15-2300/checkpoints"
        model_49 = "models/vdp1/vdp1_losssteps=49_ninter=2_loss=mse/vdp1_losssteps=49_ninter=2_loss=mse_01-15-2324/checkpoints"
        model = load_odeon_model_from_checkpoint(model_49, epoch=None)
        print(vdp_task_1(model, Path("experiments/vdp1/data") / "0", plot=True))
        for i in range(100):
            model = load_odeon_model_from_checkpoint(model_49, epoch=201+i)
            print(201+i, vdp_task_1(model, Path("experiments/vdp1/data") / "0", plot=False))
            # Best epoch: 232 with 0.025422130243601355
            pass
        #N_times_VDP_task_1(model, data_path=Path("experiments/vdp1/data"), N_datasets=50, plot=False, verbose=False)

    # config_noise_1.yaml
    if False:
        model = load_odeon_model_from_checkpoint("models/vdp1/vdp1_losssteps=15_ninter=16_loss=mse/vdp1_losssteps=15_ninter=16_loss=mse_01-15-2012/checkpoints")
        print(vdp_task_1(model, Path("experiments/vdp1/data") / "1", plot=True))
        for i in range(100):
            model = load_odeon_model_from_checkpoint("models/vdp1/vdp1_losssteps=15_ninter=16_loss=mse/vdp1_losssteps=15_ninter=16_loss=mse_01-15-2012/checkpoints", epoch=201+i)
            print(201+i, vdp_task_1(model, Path("experiments/vdp1/data") / "1", plot=False))
            # Best epoch: 236 with 0.011938288171650692
        #N_times_VDP_task_2(model, data_path=Path("experiments/vdp1/data"), N_datasets=50, plot=False, verbose=False)
    
    # config_noise_2.yaml
    if False:
        model = load_odeon_model_from_checkpoint("models/vdp1/vdp1_losssteps=15_ninter=16_loss=mse/vdp1_losssteps=15_ninter=16_loss=mse_01-15-2039/checkpoints")
        print(vdp_task_1(model, Path("experiments/vdp1/data") / "2", plot=True))
        for i in range(100):
            model = load_odeon_model_from_checkpoint("models/vdp1/vdp1_losssteps=15_ninter=16_loss=mse/vdp1_losssteps=15_ninter=16_loss=mse_01-15-2039/checkpoints", epoch=201+i)
            print(201+i, vdp_task_1(model, Path("experiments/vdp1/data") / "2", plot=False))
            # Best epoch: 265 with 0.08546633320587388
        #N_times_VDP_task_2(model, data_path=Path("experiments/vdp1/data"), N_datasets=50, plot=False, verbose=False)
    
    if False:
        model = load_odeon_model_from_checkpoint("")        # TODO !
        print(vdp_task_2(model, Path("experiments/vdp2/data") / "0", plot=True))
        N_times_VDP_task_2(model, data_path=Path("experiments/vdp2/data"), N_datasets=50, plot=False, verbose=False)

    #print("%.2f" % mocap_task(model, Path("experiments/mocap/") / "mocap09long" / "data/test/5d", plot=True, dim_left=3, left_only=True, sublist=None))

    quit()
    ckpt_path = Path("models/mocap39short/mocap39short_nsteps=25_ninter=3_loss=mse/mocap39short_nsteps=25_ninter=3_loss=mse_01-14-0409/checkpoints")
    model = load_odeon_model_from_checkpoint(ckpt_path, epoch=None)

    print("%.2f" % mocap_task(model, Path("experiments/mocap/") / "mocap39long" / "data/test/5d", plot=True, dim_left=3, left_only=True, sublist=None))

    """
    task_name = "mocap39short"
    for epoch in range(201,250):
        model = load_odeon_model_from_checkpoint(ckpt_path, epoch=epoch)
        print("%.2f" % mocap_task(model, Path("experiments/mocap/") / task_name / "data/test/5d", plot=False, dim_left=3, left_only=True, sublist=None))
    """
    # wait which is correct now... do we evaluate the NN on train and then judge it on test?

    #all_mocap_experiments(model, Path("data/mocap"), plot=True, dim_left=3, left_only=True)

    #print( vdp_task_1(model, Path("data/vanderpol/task1_uniform_grid/0"), plot=True) )
    #print( fhn_task(model, Path("data/fitzhughnagumo/missing_qudrants/0"), plot=True) )
    #print( mocap_task(model, Path("data/mocap/subject_35/short"), plot=False, dim_left=3, left_only=True) )


    #N_times_VDP_task_1(model, Path("data/vanderpol/task1_uniform_grid"), N_datasets=50, plot=False, verbose=False)
    #N_times_VDP_task_2(model, Path("data/vanderpol/task2_non_uniform_grid"), N_datasets=50, plot=False, verbose=False)
    
    #N_times_FHN_task(model, Path("data/fitzhughnagumo/missing_qudrants"), N_datasets=50, fine_MSE=False, plot=False, verbose=True)


    # really for mocap we should supply a model_left and a model_right

    """
    results_dict = {n: [] for n in range(1,17)}
    averages = {n: None for n in range(1,17)}
    medians = {n: None for n in range(1,17)}
    stds = {n: None for n in range(1,17)}
    print()
    for n_ctx_trajs in range(1,17):
        if n_ctx_trajs in [1,15,16]:
            for sublist in combinations(range(0,16), n_ctx_trajs):
                results_dict[n_ctx_trajs].append( mocap_task(model, Path("data/mocap/subject_35/long"), plot=False, dim_left=3, left_only=True, sublist=sublist) )
        else:
            for i in range(20):
                sublist = random.sample(range(0,16), n_ctx_trajs)
                results_dict[n_ctx_trajs].append( mocap_task(model, Path("data/mocap/subject_35/long"), plot=False, dim_left=3, left_only=True, sublist=sublist) )

        print(f"Performance for {n_ctx_trajs} context trajectories ({len(results_dict[n_ctx_trajs])} sublists):")

        medians[n_ctx_trajs] = statistics.median(results_dict[n_ctx_trajs])
        print(f"Median {medians[n_ctx_trajs]:.2f}")
        
        results_dict[n_ctx_trajs].sort()
        if n_ctx_trajs != 16:
            print(f"2nd lowest: {results_dict[n_ctx_trajs][1]:.2f}, 2nd highest: {results_dict[n_ctx_trajs][-2]:.2f}")
            averages[n_ctx_trajs] = statistics.mean(results_dict[n_ctx_trajs]) if n_ctx_trajs != 16 else results_dict[n_ctx_trajs][0]
            stds[n_ctx_trajs] = statistics.stdev(results_dict[n_ctx_trajs]) if n_ctx_trajs != 16 else 0
            print(f"{averages[n_ctx_trajs]:.2f} +- {stds[n_ctx_trajs]:.2f}")
        
        print([float(f"{res:.0f}") for res in results_dict[n_ctx_trajs]])
        print()
    """
    """
    WHY AM I EVALUATING MOCAP ON THE TRAINING TRAJECTORIES????
    """

    
