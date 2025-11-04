from typing import Any
from scipy.stats import ttest_1samp
import pandas as pd
import numpy as np


def get_closest(points: np.ndarray, point: np.ndarray, amount: int) -> np.ndarray:
    """
    Gets the indecies n closest points to the input points in the input points array. Not necessarily in closest to farthest order.

    Args:
        points (np.ndarray): A list of 2-dimensional points. These are the points which the input point is checked against.
        point (np.ndarray): A 2-dimensional point. This is the point which is being checked.
        amount (int): The number of points to return

    Returns:
        np.ndarray: the n closest points to the input point, not necessarily in closest to farthest order.
    """
    return np.argpartition(np.linalg.norm(points - point, axis=1), amount)[:amount]


def convert_model(
    model: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    """
    Converts an input model taken from the color model `.csv` file and a helpful dict to use in calculations

    Args:
        model (DataFrame): The DataFrame which can be directly loaded from the color model's minimized `.csv` file

    Returns:
        dict[str, dict[str, Any]]: Contains the following data
            For "optimal", "suboptimal", and "natural" has a dictionary with the following keys:
            - "points": All the points of encoders of that type
            - "check": All the points to check against in the neighbor comparison
            - "check_convexities": A dict with the quasi-convexity of the q(m|w) and q(u|w) distributions for the encoders in "check"
            - "point_convexities": A dict with the quasi-convexity of the q(m|w) and q(u|w) distributions for the encoders in "points"
    """
    output = {
        "optimal": {
            "points": [],
            "check": [],
            "check_convexities": {"qmw": [], "quw": []},
            "point_convexities": {"qmw": [], "quw": []},
        },
        "suboptimal": {
            "points": [],
            "check": [],
            "check_convexities": {"qmw": [], "quw": []},
            "point_convexities": {"qmw": [], "quw": []},
        },
        "natural": {
            "points": [],
            "check": [],
            "check_convexities": {"qmw": [], "quw": []},
            "point_convexities": {"qmw": [], "quw": []},
        },
    }

    for _, row in model.iterrows():
        for k in output.keys():
            if k == row["type"]:
                output[k]["points"].append([row["complexity"], row["accuracy"]])
                output[k]["point_convexities"]["quw"].append(row["convexity-quw"])
                output[k]["point_convexities"]["qmw"].append(row["convexity-qmw"])
            if k != row["type"] or row["type"] == "suboptimal":
                output[k]["check"].append([row["complexity"], row["accuracy"]])
                output[k]["check_convexities"]["quw"].append(row["convexity-quw"])
                output[k]["check_convexities"]["qmw"].append(row["convexity-qmw"])

    for k in output.keys():
        output[k]["points"] = np.array(output[k]["points"])
        output[k]["check"] = np.array(output[k]["check"])

    return output


def get_neighbor_comparison(amount: int, model: pd.DataFrame):
    """
    Takes in a number of neighbors to compare to and the color model from the `.csv` file and displays information about neighboring encoders

    Args:
        amount (int): The number of neighbors to compare to.
        model (DataFrame): The DataFrame which can be directly loaded from the color model's minimized `.csv` file.
    """
    converted_model = convert_model(model)

    comparison_qmw = {"optimal": [], "suboptimal": [], "natural": []}
    comparison_quw = {"optimal": [], "suboptimal": [], "natural": []}
    for k in converted_model.keys():
        suboptimal = k == "suboptimal"
        for i, p in enumerate(converted_model[k]["points"]):
            higher_than_qmw = -0.5 if suboptimal else 0
            higher_than_quw = -0.5 if suboptimal else 0
            if suboptimal:
                # Check includes itself so get 11 instead of 10
                closest = get_closest(converted_model[k]["check"], p, amount + 1)
            else:
                closest = get_closest(converted_model[k]["check"], p, amount)
            for c in closest:
                qmw_diff = (
                    converted_model[k]["point_convexities"]["qmw"][i]
                    - converted_model[k]["check_convexities"]["qmw"][c]
                )
                quw_diff = (
                    converted_model[k]["point_convexities"]["quw"][i]
                    - converted_model[k]["check_convexities"]["quw"][c]
                )
                if qmw_diff > 0:
                    higher_than_qmw += 1
                if qmw_diff == 0:
                    higher_than_qmw += 0.5
                if quw_diff > 0:
                    higher_than_quw += 1
                if quw_diff == 0:
                    higher_than_quw += 0.5
            comparison_qmw[k].append(higher_than_qmw / amount)
            comparison_quw[k].append(higher_than_quw / amount)

    total_qmw = (
        comparison_qmw["natural"]
        + comparison_qmw["suboptimal"]
        + comparison_qmw["optimal"]
    )
    print("Average comparison (q(m|w)):", sum(total_qmw) / len(total_qmw))
    print(
        "Average natural language (q(m|w)):",
        sum(comparison_qmw["natural"]) / len(comparison_qmw["natural"]),
    )
    print(
        "Average optimal encoder (q(m|w)):",
        sum(comparison_qmw["optimal"]) / len(comparison_qmw["optimal"]),
    )
    print(
        "Average suboptimal encoder (q(m|w)):",
        sum(comparison_qmw["suboptimal"]) / len(comparison_qmw["suboptimal"]),
    )
    print()

    total_quw = (
        comparison_quw["natural"]
        + comparison_quw["suboptimal"]
        + comparison_quw["optimal"]
    )
    print("Average comparison (q(u|w)):", sum(total_quw) / len(total_quw))
    print(
        "Average natural language (q(u|w)):",
        sum(comparison_quw["natural"]) / len(comparison_quw["natural"]),
    )
    print(
        "Average optimal encoder (q(u|w)):",
        sum(comparison_quw["optimal"]) / len(comparison_quw["optimal"]),
    )
    print(
        "Average suboptimal encoder (q(u|w)):",
        sum(comparison_quw["suboptimal"]) / len(comparison_quw["suboptimal"]),
    )


def check_difference_significance(amount: int, model: pd.DataFrame):
    """
    Takes in a number of neighbors to compare to and the color model from the `.csv` file and finds correlation between
    the quasi-convexity of a natural language's q(m|w) and q(u|w) distributions and that of their `amount` closest neighbors

    Args:
        amount (int): The number of neighbors to compare to.
        model (DataFrame): The DataFrame which can be directly loaded from the color model's minimized `.csv` file.
    """
    converted = convert_model(model)

    comparison_qmw = []
    comparison_quw = []

    for i, p in enumerate(converted["natural"]["points"]):
        closest = get_closest(converted["natural"]["check"], p, amount)
        for c in closest:
            comparison_qmw.append(
                converted["natural"]["point_convexities"]["qmw"][i]
                - converted["natural"]["check_convexities"]["qmw"][c]
            )
            comparison_quw.append(
                converted["natural"]["point_convexities"]["quw"][i]
                - converted["natural"]["check_convexities"]["quw"][c]
            )

    print(ttest_1samp(comparison_qmw, 0))
    print(ttest_1samp(comparison_quw, 0))


if __name__ == "__main__":
    model = pd.read_csv("./colors/data/minimized/color_model.csv", header=0)

    check_difference_significance(10, model)
    get_neighbor_comparison(10, model)
