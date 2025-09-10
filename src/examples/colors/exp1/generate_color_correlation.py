import numpy as np
import pandas as pd

from ..utils.correlation import calc_corr_to_tuple, csv_to_latex


def calculate_correlations(
    qmw: bool,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """
    Calculates the Person correlation coefficients between the optimality, complexity, and accuracy (independent variables)
    and the quasi-convexity of encoders (dependent variable). Can be used to check the quasi-convexity of q(m | w) or q(u | w)
    distributions

    Args:
        qmw (bool): If True, the correlations between the independent variables and the quasi-convexity q(w | u)

    Returns:
        tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
        The correlation coefficients and p values for the following dependent variables (in order):
        - optimality
        - complexity
        - accuracy
    """
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
    optimality = np.array(model["optimality"])

    opt_coeff = calc_corr_to_tuple(
        optimality,
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
    """
    Creates the correlation coefficients and p values for the color values and save it to a file.
    Does this for the quasi-convexity of the q(m|w) and q(u|w) distributions and optimality, complexity, and accuracy.
    """
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


if __name__ == "__main__":
    create_table()
    print(csv_to_latex("color_coefficients"))
