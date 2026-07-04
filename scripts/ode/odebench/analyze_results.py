import json
from pathlib import Path


def r2_threshold_analysis():
    with open(Path("experiments/odebench/results/new174/all_r2_stats.json"), "r") as f:
        d = json.load(f)["new_pretraining_1_continued_ckpt-None"]

        recon_r2_above_09 = {rho: {} for rho in d["reconstruction"].keys()}
        gener_r2_above_09 = {rho: {} for rho in d["generalization"].keys()}
        gener_r2_above_08 = {rho: {} for rho in d["generalization"].keys()}

        for task in d.keys():
            print(task)  # reconstruction, generalization
            for rho in d[task].keys():
                for sigma in d[task][rho].keys():
                    print(f"rho: {rho}")  # 0.0, 0.5
                    print(f"sigma: {sigma}")  # 0.0, 0.03, 0.05
                    # print(d[task][rho][sigma])
                    r2_list = d[task][rho][sigma]
                    # r2_mean = np.mean(r2_list)
                    # r2_std = np.std(r2_list)
                    # print(f"R2 mean: {r2_mean}")
                    # print(f"R2 std: {r2_std}")

                    r2_above09 = 0
                    r2_above08 = 0
                    for r2 in r2_list:
                        if r2 > 0.9:
                            r2_above09 += 1
                        if r2 > 0.8:
                            r2_above08 += 1

                    if task == "reconstruction":
                        print(rho == 0.0)
                        print(recon_r2_above_09[rho])
                        recon_r2_above_09[rho][sigma] = r2_above09 / len(r2_list)
                    if task == "generalization":
                        gener_r2_above_09[rho][sigma] = r2_above09 / len(r2_list)
                        gener_r2_above_08[rho][sigma] = r2_above08 / len(r2_list)
                    print(f"R2 above 0.9: {r2_above09 / len(r2_list):.1%}")
                    if task == "generalization":
                        print(f"R2 above 0.8: {r2_above08 / len(r2_list):.1%}")
                    print()
                print()
            print()

    for ddd in [recon_r2_above_09, gener_r2_above_09, gener_r2_above_08]:
        string = ""
        for rho in d[task].keys():
            for sigma in d[task][rho].keys():
                string += f"& {ddd[rho][sigma] * 100:.1f}\\% "
        print(string)
        print()


def curves():
    pass


if __name__ == "__main__":
    r2_threshold_analysis()
