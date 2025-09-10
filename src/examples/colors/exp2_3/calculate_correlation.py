from ..utils.correlation import csv_to_latex, calc_corr_to_tuple

import numpy as np
import pandas as pd


def calculate_correlations(
    name: str, qmw: bool = True
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """
    Calculates the Person correlation coefficients between the optimality, complexity, and accuracy (independent variables)
    and the quasi-convexity of encoders (dependent variable). Can be used to check the quasi-convexity of q(m | w) or q(u | w)
    distributions

    Args:
        name (str): The name of the model (without the `model-` prefix or the file extension) to load
        qmw (bool): If True, the correlations between the independent variables and the quasi-convexity q(w | u)

    Returns:
        tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
        The correlation coefficients and p values for the following dependent variables (in order):
        - optimality
        - complexity
        - accuracy
    """
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
    optimality = np.array(model["optimality"])

    opt_coeff = calc_corr_to_tuple(
        optimality,
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


def create_table(names: list[str], titles: list[str], file_name: str, qmw: bool = True):
    """
    Creates the correlation coefficients and p values for the input models and saves them to a file.
    Does this for the quasi-convexity of the q(m|w) or q(u|w) distribution (based on the qmw arg)
    and optimality, complexity, and accuracy.

    Args:
        names (list[str]): The list of models to load and calculate correlations
        titles (list[str]): The display names for each model (the leftmost column of the table)
        file_names (str): The file to save to (without the file extension)
        qmw (bool): Whether to use the quasi-convexity of the q(m|w) or q(u|w) distribution
    """
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
        opt_coeff, comp_coeff, iwu_coeff = calculate_correlations(name, qmw=qmw)
        coeffs["opt"].append(opt_coeff[0])
        coeffs["p opt"].append(opt_coeff[1])
        coeffs["I(M; W)"].append(comp_coeff[0])
        coeffs["p I(M; W)"].append(comp_coeff[1])
        coeffs["I(W; U)"].append(iwu_coeff[0])
        coeffs["p I(W; U)"].append(iwu_coeff[1])

    df = pd.DataFrame(data=coeffs)
    df.to_csv(f"./colors/output/convexity/{file_name}.csv", index=False)


if __name__ == "__main__":
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
