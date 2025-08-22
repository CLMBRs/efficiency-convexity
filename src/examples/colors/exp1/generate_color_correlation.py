from scipy.stats import pearsonr
import numpy as np
import pandas as pd

from ..utils.correlation import csv_to_latex


def find_frontier_optimality(frontier, point):
    return -np.min(np.linalg.norm(frontier - point, axis=1))


def calc_corr_to_tuple(x, y):
    corr = pearsonr(x, y)
    c = round(corr.statistic, 3)
    p = round(corr.pvalue, 3)
    return (c, p)


def calculate_correlations(qmw):
    model = pd.read_csv("./colors/data/minimized/color_model.csv", header=0)

    frontier = []
    suboptimal = []
    natural = []

    frontier_convexity = []
    other_convexity = []

    for _, row in model.iterrows():
        if row["type"] == "optimal":
            frontier.append([row["complexity"], row["accuracy"]])
            frontier_convexity.append(row[f"convexity-q{'m' if qmw else 'u'}w"])
        if row["type"] == "suboptimal":
            suboptimal.append([row["complexity"], row["accuracy"]])
            other_convexity.append(row[f"convexity-q{'m' if qmw else 'u'}w"])
        if row["type"] == "natural":
            natural.append([row["complexity"], row["accuracy"]])
            other_convexity.append(row[f"convexity-q{'m' if qmw else 'u'}w"])

    frontier = np.array(frontier)
    suboptimal = np.array(suboptimal)
    natural = np.array(natural)
    optimality = []

    for s in suboptimal:
        optimality.append(find_frontier_optimality(frontier, s))
    for n in natural:
        optimality.append(find_frontier_optimality(frontier, n))

    optimality = np.array(optimality)

    opt_coeff = calc_corr_to_tuple(
        np.concatenate((np.zeros(len(frontier)), optimality)),
        np.concat((frontier_convexity, other_convexity)),
    )
    comp_coeff = calc_corr_to_tuple(
        np.concatenate((frontier[:, 0], suboptimal[:, 0], natural[:, 0])),
        np.concat((frontier_convexity, other_convexity)),
    )
    iwu_coeff = calc_corr_to_tuple(
        np.concatenate((frontier[:, 1], suboptimal[:, 1], natural[:, 1])),
        np.concat((frontier_convexity, other_convexity)),
    )
    return opt_coeff, comp_coeff, iwu_coeff


def create_table():
    coeffs = {
        "env": ["$q(m|w)$", "$q(u|w)$"],
        "opt": [],
        "p opt": [],
        "I(M; W)": [],
        "p I(M; W)": [],
        "I(W; U)": [],
        "p I(W; U)": [],
    }

    for i in [True, False]:
        opt_coeff, comp_coeff, iwu_coeff = calculate_correlations(i)
        coeffs["opt"].append(opt_coeff[0])
        coeffs["p opt"].append(opt_coeff[1])
        coeffs["I(M; W)"].append(comp_coeff[0])
        coeffs["p I(M; W)"].append(comp_coeff[1])
        coeffs["I(W; U)"].append(iwu_coeff[0])
        coeffs["p I(W; U)"].append(iwu_coeff[1])

    df = pd.DataFrame(data=coeffs)
    df.to_csv(f"./colors/output/convexity/color_coefficients.csv", index=False)


create_table()
print(csv_to_latex("color_coefficients"))
