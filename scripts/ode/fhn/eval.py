from experiments_on_vdp_fhn_mocap import vdp_task_1, vdp_task_2, fhn_task, mocap_task_new
from utils.eval_models import OdeonEval, PredictionModel
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path

def finetuning_report(model_path: str, filename: str = "experiments/vdp1/finetuning_report.pdf", fine_MSE: bool = False):

    with PdfPages(filename) as pdf:
        model = OdeonEval(Path(model_path)/"checkpoints")
        
        fig, ax = plt.subplots(1,2, figsize=(20,8))
        score = fhn_task(model, Path("experiments/fhn/data_gpode"), plot=ax[0], plot_type="1D", fine_MSE=fine_MSE)
        fhn_task(model, Path("experiments/fhn/data_gpode"), plot=ax[1], plot_type="2D", fine_MSE=fine_MSE)
        print(score)
        ax[0].set_title(f"FHN Best Model (Epoch {model.epoch}): {score:.3f}")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        for epoch in range(201, 301, 5):
            fig, ax = plt.subplots(1, 2, figsize=(16, 8))
            model = OdeonEval(Path(model_path) / "checkpoints", epoch=epoch)
            score = fhn_task(model, Path("experiments/fhn/data_gpode"), plot=ax[0], plot_type="1D", fine_MSE=fine_MSE)
            fhn_task(model, Path("experiments/fhn/data_gpode"), plot=ax[1], plot_type="2D", fine_MSE=fine_MSE)
            print(score)
            fig.suptitle(f"FHN Epoch {epoch}: {score:.3f}", fontsize=14)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
    


def print_all_epochs(model_path: str, fine_MSE: bool = False):
    for epoch in range(201,1001,1):
        model = OdeonEval(Path(model_path)/"checkpoints", epoch=epoch)
        print(epoch, fhn_task(model, Path("experiments/fhn/data_gpode"), plot=False, fine_MSE=fine_MSE))


if __name__ == "__main__":
    for model_path in [
        #"models/fhn/gridsearch_01-28-1636",
        #"models/fhn/gridsearch_01-28-1637",
        #"models/fhn/gridsearch_01-28-1650",
        #"models/fhn/gridsearch_01-28-1651",
        #"models/fhn/gridsearch_01-28-1716",
        #"models/fhn/gridsearch_01-28-1717",
        #"models/fhn/gridsearch_01-28-1846",
        #"models/fhn/gridsearch_01-28-1939",
        "models/fhn/gridsearch_01-28-2113",
        "models/fhn/gridsearch_01-28-2120",
    ]:
        model = OdeonEval(Path(model_path)/"checkpoints")
        print(fhn_task(model, Path("experiments/fhn/data_gpode"), plot=False, fine_MSE=True))
        #print_all_epochs(model_path)
        #print(vdp_task_1(model, Path("experiments/vdp1/data_gpode"), plot=False))
        #print(vdp_task_2(model, Path("experiments/vdp2/data_gpode"), plot=False))
        #print(mocap_task_new(model, Path("experiments/mocap/mocap09short/data/test/5d"), plot=False, dim_left=3, left_only=True))
        #finetuning_report(model_path, filename=f"experiments/fhn/report_{model_path.split('/')[-1].split('_')[-1]}.pdf", fine_MSE=False)
