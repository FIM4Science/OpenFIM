from experiments_on_vdp_fhn_mocap import vdp_task_1, vdp_task_2, fhn_task, mocap_task_new
from utils.eval_models import OdeonEval, PredictionModel
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path

def finetuning_report(model_path: str, filename: str = "experiments/vdp1/finetuning_report.pdf"):
    #model = OdeonEval(Path(model_path)/"checkpoints")
    #score = vdp_task_1(model, Path("experiments/vdp1/data_gpode"), plot=False)
    #print(score)

    with PdfPages(filename) as pdf:
        for epoch in range(201,251,1):
            fig, ax = plt.subplots(figsize=(8,8))
            model = OdeonEval(Path(model_path)/"checkpoints", epoch=epoch)
            score = vdp_task_1(model, Path("experiments/vdp1/data_gpode"), plot=ax, plot_type="2D")
            print(score)
            ax.set_title(f"VDP1 Epoch {epoch}: {score:.3f}")
            
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
    


def print_all_epochs(model_path: str):
    #model = OdeonEval(Path(model_path)/"checkpoints")
    #score = vdp_task_1(model, Path("experiments/vdp1/data_gpode"), plot=False)
    #print(score)

    for epoch in range(201,1001,1):
        model = OdeonEval(Path(model_path)/"checkpoints", epoch=epoch)
        print(epoch, vdp_task_1(model, Path("experiments/vdp1/data_gpode"), plot=False))


if __name__ == "__main__":
    model_path = "models/vdp1/gridsearch_01-28-0955"
    model = OdeonEval(Path(model_path)/"checkpoints")
    print(vdp_task_1(model, Path("experiments/vdp1/data_gpode"), plot=False))
    print_all_epochs(model_path)
    #finetuning_report(model_path, filename="experiments/vdp1/reportreport.pdf")

    quit()
    model_path = Path("models/vdp1/gridsearch_01-28-0045")
    model = OdeonEval(Path(model_path)/"checkpoints")
    print(vdp_task_1(model, Path("experiments/vdp1/data_gpode"), plot=False))

    quit()
    model_path="/cephfs/users/huebers/FIM/models/vdp1/gridsearch_01-27-2311"
    finetuning_report(model_path, filename="experiments/vdp1/finetuning_report_final_points.pdf")

    for epoch in range(201,301,1):
        model = OdeonEval(Path(model_path)/"checkpoints", epoch=epoch)
        print(epoch, vdp_task_1(model, Path("experiments/vdp1/data_gpode"), plot=False))

    quit()
    vdp1 = [
        "/cephfs/users/huebers/FIM/models/vdp1/gridsearch_01-27-2311",
        "models/vdp1/gridsearch-model-train_config-step_noise_scale-0-0_01-27-2032",
        "models/vdp1/gridsearch-model-train_config-step_noise_scale-0-1_01-27-2103",
    ]
    model = OdeonEval(Path(vdp1[0])/"checkpoints")
    print(vdp_task_1(model, Path("experiments/vdp1/data_gpode"), plot=False))

    # experiment = vdp1[0]
    # for i in range(201,301,10):
    #     model = OdeonEval(Path(experiment)/"checkpoints", epoch=i)
    #     print(i, vdp_task_1(model, Path("experiments/vdp1/data_gpode"), plot=False))

    quit()

    for experiment in vdp1:
        model = OdeonEval(Path(experiment)/"checkpoints", epoch=None)
        print(vdp_task_1(model, Path("experiments/vdp1/data_gpode"), plot=True))

    vdp2 = [
        "models/vdp2/gridsearch-model-train_config-step_noise_scale-0-0_01-27-2050",
        "models/vdp2/gridsearch-model-train_config-step_noise_scale-0-1_01-27-2121",
        "models/vdp2/gridsearch-model-train_config-step_noise_scale-0-5_01-27-2151"
    ]

    for experiment in vdp2:
        model = OdeonEval(Path(experiment)/"checkpoints", epoch=None)
        print(vdp_task_2(model, Path("experiments/vdp2/data_gpode"), plot=True))
    
    fhn = [
        "models/fhn/gridsearch-model-train_config-step_noise_scale-0-0_01-27-2053",
        "models/fhn/gridsearch-model-train_config-step_noise_scale-0-1_01-27-2106",
        "models/fhn/gridsearch-model-train_config-step_noise_scale-0-5_01-27-2119"
    ]

    for experiment in fhn:
        model = OdeonEval(Path(experiment)/"checkpoints", epoch=None)
        print(fhn_task(model, Path("experiments/fhn/data_gpode"), fine_MSE=False, plot=True))
