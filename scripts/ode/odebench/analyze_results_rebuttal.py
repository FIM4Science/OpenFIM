import json
from pathlib import Path


# rho=p, sigma=n
# model name - reconstruction/generalization - p=0.0,0.5 - n=0.0,0.03,0.05

# there are 61 systems in total. 23 are 1D, 28 (24-51) are 2D, 10 (52-61) are 3D.

polynomial_indices = [
    1,
    2,
    3,
    5,
    6,
    8,
    9,
    11,
    12,
    16,
    17,
    24,
    25,
    26,
    27,
    29,
    30,
    31,
    37,
    38,
    39,
    40,
    41,
    45,
    49,
    50,
    52,
    54,
    55,
    56,
    57,
    58,
    59,
    60,
    61,
    63,
]
# this one contains 9/10 3D systems and 15/28 2D systems and 11/23 1D systems
# 31.4% are 1D, 42.9% are 2D, 25.7% are 3D

non_polynomial_indices = [4, 7, 10, 13, 14, 15, 18, 19, 20, 21, 22, 23, 28, 32, 33, 34, 35, 36, 42, 43, 44, 46, 47, 48, 51, 53, 62]
# this one contains 1/10 3D systems, 13/28 2D systems and 12/23 1D systems
# 46.2% are 1D, 46.4% are 2D, 3.8% are 3D

all_indices = range(1, 62)

# idxs = polynomial_indices
idxs = non_polynomial_indices
# idxs = all_indices

range_1d = range(1, 24)
range_2d = range(24, 52)
range_3d = range(52, 62)

idx_range = range_2d


def r2_threshold_analysis():
    # with open(Path("experiments/odebench/results/max/all_r2_stats.json"), "r") as f:
    # d = json.load(f)["base_model_ckpt-None"]
    with open(Path("experiments/odebench/results/odeformer/all_r2_stats.json"), "r") as f:
        d = json.load(f)["odeformer"]

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
                    r2_list = [r2_list[2 * (i - 1)] for i in idxs if i in idx_range] + [r2_list[2 * i - 1] for i in idxs if i in idx_range]
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
