from ..utils.correlation import csv_to_latex

from scipy.stats import pearsonr
import numpy as np
import pandas as pd


def find_frontier_optimality(frontier, point):
    return -np.min(np.linalg.norm(frontier - point, axis=1))


def calc_corr_to_tuple(x, y):
    corr = pearsonr(x, y)
    c = round(corr.statistic, 3)
    p = round(corr.pvalue, 3)
    return (c, p)


def calculate_correlations(name, qmw=True):
    model = pd.read_csv(f"./colors/data/minimized/model-{name}.csv", header=0)

    frontier = []
    suboptimal = []

    frontier_convexity = []
    suboptimal_convexity = []

    for _, row in model.iterrows():
        if row["type"] == "optimal":
            frontier.append([row["complexity"], row["accuracy"]])
            frontier_convexity.append(row[f"convexity-q{'m' if qmw else 'u'}w"])
        if row["type"] == "suboptimal":
            suboptimal.append([row["complexity"], row["accuracy"]])
            suboptimal_convexity.append(row[f"convexity-q{'m' if qmw else 'u'}w"])

    frontier = np.array(frontier)
    suboptimal = np.array(suboptimal)
    optimality = []

    for s in suboptimal:
        optimality.append(find_frontier_optimality(frontier, s))

    optimality = np.array(optimality)

    opt_coeff = calc_corr_to_tuple(
        np.concatenate((np.zeros(len(frontier)), optimality)),
        np.concat((frontier_convexity, suboptimal_convexity)),
    )
    comp_coeff = calc_corr_to_tuple(
        np.concatenate((frontier[:, 0], suboptimal[:, 0])),
        np.concat((frontier_convexity, suboptimal_convexity)),
    )
    iwu_coeff = calc_corr_to_tuple(
        np.concatenate((frontier[:, 1], suboptimal[:, 1])),
        np.concat((frontier_convexity, suboptimal_convexity)),
    )
    return opt_coeff, comp_coeff, iwu_coeff


def create_table(names, titles, file_name, qmw=True):
    coeffs = {
        "env": [f"\\textsc{'{' + t + '}'}" for t in titles],
        "opt": [],
        "p opt": [],
        "I(M; W)": [],
        "p I(M; W)": [],
        "I(W; U)": [],
        "p I(W; U)": [],
    }

    for name in names:
        opt_coeff, comp_coeff, iwu_coeff = calculate_correlations(
            name, qmw=qmw
        )
        coeffs["opt"].append(opt_coeff[0])
        coeffs["p opt"].append(opt_coeff[1])
        coeffs["I(M; W)"].append(comp_coeff[0])
        coeffs["p I(M; W)"].append(comp_coeff[1])
        coeffs["I(W; U)"].append(iwu_coeff[0])
        coeffs["p I(W; U)"].append(iwu_coeff[1])

    df = pd.DataFrame(data=coeffs)
    df.to_csv(f"./colors/output/convexity/{file_name}.csv", index=False)


create_table(
    [
        "cpum",
        "npum",
        "cpdm",
        "npdm",
        "cpum_dual",
        "npum_dual",
        "cpdm_dual",
        "npdm_dual",
        "cpdm_convex",
        "cpdm_adj",
        "npdm_adj",
        "manhattan_5_5",
    ],
    [
        "CPUM",
        "NPUM",
        "CPDM",
        "NPDM",
        "CPUM-Dual",
        "NPUM-Dual",
        "CPDM-Dual",
        "NPDM-Dual",
        "CPDM-Convex",
        "CPDM-Adj",
        "NPDM-Adj",
        "Manhattan-5-5",
    ],
    "qmw_coefficients",
)
create_table(
    [
        "cpum",
        "npum",
        "cpdm",
        "npdm",
        "cpum_dual",
        "npum_dual",
        "cpdm_dual",
        "npdm_dual",
        "npdm_shift",
        "cpum_split",
        "cpdm_split",
        "manhattan_5_5",
    ],
    [
        "CPUM",
        "NPUM",
        "CPDM",
        "NPDM",
        "CPUM-Dual",
        "NPUM-Dual",
        "CPDM-Dual",
        "NPDM-Dual",
        "NPDM-Shift",
        "CPUM-Split",
        "CPDM-Split",
        "Manhattan-5-5",
    ],
    "quw_coefficients",
    qmw=False,
)

print(csv_to_latex("qmw_coefficients"))
print(csv_to_latex("quw_coefficients"))
