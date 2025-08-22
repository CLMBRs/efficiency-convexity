import statsmodels.formula.api as smf
import pandas as pd
import numpy as np
import pickle
import sys


def find_frontier_optimality(frontier, point):
    return -np.min(np.linalg.norm(frontier - point, axis=1))


def ols_from_model(model):

    frontier = []
    all_points = []

    for _, row in model.iterrows():
        if row["type"] == "optimal":
            frontier.append([row["complexity"], row["accuracy"]])
        all_points.append([row["complexity"], row["accuracy"]])

    frontier = np.array(frontier)
    all_points = np.array(all_points)
    optimality = []

    for p in all_points:
        optimality.append(find_frontier_optimality(frontier, p))

    model["optimality"] = optimality

    model = model.rename(
        columns={"convexity-qmw": "convexity_qmw", "convexity-quw": "convexity_quw"}
    )

    results = smf.ols(
        "convexity_qmw ~ optimality + complexity * accuracy", data=model
    ).fit()
    print(results.summary())

    results = smf.ols(
        "convexity_quw ~ optimality + complexity * accuracy", data=model
    ).fit()
    print(results.summary())


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise ValueError("Invalid number of arguments, wanted 1")

    model = pd.read_csv(f"./colors/data/minimized/{sys.argv[1]}.csv", header=0)

    ols_from_model(model)
