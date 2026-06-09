from experiments_on_vdp_fhn_mocap import vdp_task_1, vdp_task_2, fhn_task, mocap_task_new
from utils.eval_models import OdeonEval, PredictionModel
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path


JUST_EVALUATE_BEST = False

JUST_EVALUATE_EPOCH = None


fhn_paths = [
    #"models/fhn/fhn_01-29-0710",   # 0.0765 step_noise_scale: 0.0, 800 epochs
    #"models/fhn/fhn_01-29-0711",   # 0.0514 step_noise_scale: 0.2, 800 epochs
    #"models/fhn/gridsearch_01-29-1004",  # l1 200 epochs
    "models/fhn/gridsearch_01-29-1048",  # the first submission version
]
fhn_data_path = Path("experiments/fhn/data_gpode")

vdp1_paths = [
        #"models/vdp1/vdp1_01-29-0215",   # 0.31328753888874056
        #"models/vdp1/vdp1_01-29-0217",   # 0.9670021602476649   step_noise_scale: 0.0, 200 epochs
        #"models/vdp1/vdp1_01-29-0218",   # 0.17629440836742574   step_noise_scale: 0.2, 200 epochs
        #"models/vdp1/vdp1_01-29-0612",   # 0.7840431497805289
        # L1:
        "models/vdp1/vdp1_01-29-0707",   # 0.1576250891565371 step_noise_scale: 0.0, 800 epochs   
        "models/vdp1/vdp1_01-29-0709",   # 0.5733501559243287 step_noise_scale: 0.2, 800 epochs
]
vdp1_data_path = Path("experiments/vdp1/data_gpode")


vdp2_paths = [
    "models/vdp2/vdp2_01-29-0139",   # 0.26173891175248637   noise 0.0, mse, 200 epochs
    "models/vdp2/vdp2_01-29-0140",   # 0.3590333845520653    noise 0.2, mse, 200 epochs
    "models/vdp2/vdp2_01-29-0223",   # 0.25664853422809225   noise 0.0, l1, 800 epochs
    "models/vdp2/vdp2_01-29-0224",   # 0.4090886507630224    noise 0.2, l1, 800 epochs
]
vdp2_data_path = Path("experiments/vdp2/data_gpode")


mocap09long_paths = [
    #"models/vdp1/mocap09long_01-29-0311",   # 5.04
    #"models/vdp1/mocap09long_01-29-0231",
    #"models/mocap09long/mocap09long_01-29-0328",
    #"models/mocap09long/mocap09long_01-29-0609",  # 5.354363808221613
]
mocap09long_data_path = Path("experiments/mocap/mocap09long/data/train/5d")


mocap09short_paths = [
    "models/mocap09short/mocap09short_01-29-0610",   # 7.54600696040987
]
mocap09short_data_path = Path("experiments/mocap/mocap09short/data/train/5d")

mocap35short_paths = [
    "models/mocap35short/mocap35short_01-29-0614",   # 6.920406126780986
]
mocap35short_data_path = Path("experiments/mocap/mocap35short/data/train/5d")

mocap35long_paths = [
    "models/mocap35long/mocap35long_01-29-0615",   # 11.730430632700338
]
mocap35long_data_path = Path("experiments/mocap/mocap35long/data/train/5d")

mocap39short_paths = [
    "models/mocap39short/mocap39short_01-29-0618",   # 15.561364474726938
]
mocap39short_data_path = Path("experiments/mocap/mocap39short/data/train/5d")

mocap39long_paths = [
    "models/mocap39long/mocap39long_01-29-0619",   # 19.889298272404012
]
mocap39long_data_path = Path("experiments/mocap/mocap39long/data/train/5d")


# FHN
if True:
    DATA_PATH = fhn_data_path
    task = fhn_task
    model_paths = fhn_paths

# vdp1
if False:
    DATA_PATH = vdp1_data_path
    task = vdp_task_1
    model_paths = vdp1_paths

# vdp2
if False:
    DATA_PATH = vdp2_data_path
    task = vdp_task_2
    model_paths = vdp2_paths

if False:
    DATA_PATH = mocap09long_data_path
    task = lambda model, data_path, plot: mocap_task_new(model, data_path, plot=plot, dim_left=3, left_only=True) #vdp_task_2 #vdp_task_1 #lambda model, data_path, plot: mocap_task_new(model, data_path, plot=plot, dim_left=3, left_only=True)
    model_paths = mocap09long_paths


def finetuning_report(model_path: str, filename: Path = "finetuning_report.pdf"):

    with PdfPages(Path(model_path)/filename) as pdf:
        model = OdeonEval(Path(model_path)/"checkpoints")
        
        fig, ax = plt.subplots(1,1, figsize=(16,8))
        score = task(model, DATA_PATH, plot=False)
        print(score)
        ax.set_title(f"Best Model (Epoch {model.epoch}): {score:.3f}")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        for epoch in range(201, 401, 2):
            fig, ax = plt.subplots(1, 1, figsize=(16, 8))
            model = OdeonEval(Path(model_path) / "checkpoints", epoch=epoch)
            score = task(model, DATA_PATH, plot=False)
            #task(model, DATA_PATH, plot=ax, plot_type="2D")
            print(score)
            fig.suptitle(f"Epoch {epoch}: {score:.3f}", fontsize=14)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            with open(Path(model_path)/"task_loss.log", "a") as f:
                f.write(f"{epoch}: {score:.3f}\n")
    

def print_all_epochs(model_path: str):
    for epoch in range(201,1001,1):
        model = OdeonEval(Path(model_path)/"checkpoints", epoch=epoch)
        print(epoch, task(model, DATA_PATH, plot=False))


if __name__ == "__main__":
    for model_path in model_paths:
        if JUST_EVALUATE_BEST:
            model = OdeonEval(Path(model_path)/"checkpoints", epoch=None)
            print( f"Best model: {task(model, DATA_PATH, plot=False)}" )
        elif JUST_EVALUATE_EPOCH is not None:
            model = OdeonEval(Path(model_path)/"checkpoints", epoch=JUST_EVALUATE_EPOCH)
            print( f"Epoch {JUST_EVALUATE_EPOCH}: {task(model, DATA_PATH, plot=False)}" )
        else:
            finetuning_report(model_path)
