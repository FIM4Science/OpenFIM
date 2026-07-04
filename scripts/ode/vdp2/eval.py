from pathlib import Path

from experiments_on_vdp_fhn_mocap import vdp_task_2
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from utils.eval_models import OdeonEval


JUST_EVALUATE_BEST = False

JUST_EVALUATE_EPOCH = None


DATA_PATH = Path("experiments/vdp2/data_gpode")
task = vdp_task_2


def finetuning_report(model_path: str, filename: Path = "finetuning_report.pdf"):

    with PdfPages(Path(model_path) / filename) as pdf:
        model = OdeonEval(Path(model_path) / "checkpoints")

        fig, ax = plt.subplots(1, 1, figsize=(16, 8))
        score = task(model, DATA_PATH, plot=ax, plot_type="1D")
        # task(model, DATA_PATH, plot=ax[1], plot_type="2D")
        print(score)
        ax.set_title(f"VDP2 Best Model (Epoch {model.epoch}): {score:.3f}")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        for epoch in range(201, 401, 2):
            fig, ax = plt.subplots(1, 1, figsize=(16, 8))
            model = OdeonEval(Path(model_path) / "checkpoints", epoch=epoch)
            score = task(model, DATA_PATH, plot=ax, plot_type="1D")
            task(model, DATA_PATH, plot=ax, plot_type="2D")
            print(score)
            fig.suptitle(f"VDP2 Epoch {epoch}: {score:.3f}", fontsize=14)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            with open(Path(model_path) / "task_loss.log", "a") as f:
                f.write(f"{epoch}: {score:.3f}\n")


def print_all_epochs(model_path: str):
    for epoch in range(201, 1001, 1):
        model = OdeonEval(Path(model_path) / "checkpoints", epoch=epoch)
        print(epoch, task(model, DATA_PATH, plot=False))


if __name__ == "__main__":
    for model_path in [
        "models/vdp2/vdp2_01-29-0223",
        "models/vdp2/vdp2_01-29-0224",
    ]:
        if JUST_EVALUATE_BEST:
            model = OdeonEval(Path(model_path) / "checkpoints", epoch=None)
            print(f"Best model: {task(model, DATA_PATH, plot=False)}")
        elif JUST_EVALUATE_EPOCH is not None:
            model = OdeonEval(Path(model_path) / "checkpoints", epoch=JUST_EVALUATE_EPOCH)
            print(f"Epoch {JUST_EVALUATE_EPOCH}: {task(model, DATA_PATH, plot=False)}")
        else:
            finetuning_report(model_path)
