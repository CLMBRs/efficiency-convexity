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
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Converts an input model taken from the color model `.csv` file and returns helpful arrays which are used in calculations

    Args:
        model (DataFrame): The DataFrame which can be directly loaded from the color model's minimized `.csv` file

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            The following arrays, in order:
        - The points from optimal encoders
        - The points from suboptimal encoders
        - The points from natural language encoders
        - The concatination of all points (in order: optimal, suboptimal, natural)
        - The concatination of suboptimal and natural language encoder points
        - The concatination of optimal and suboptimal encoder points
        - The quasi-convexity of the q(m|w) distributions for all encoders in points
        - The quasi-convexity of the q(u|w) distributions for all encoders in points
    """
    frontier = []
    suboptimal = []
    natural = []
    convexities_qmw = []
    convexities_quw = []

    for _, row in model.iterrows():
        if row["type"] == "optimal":
            frontier.append([row["complexity"], row["accuracy"]])
        if row["type"] == "suboptimal":
            suboptimal.append([row["complexity"], row["accuracy"]])
        if row["type"] == "natural":
            natural.append([row["complexity"], row["accuracy"]])
        convexities_qmw.append(row["convexity-qmw"])
        convexities_quw.append(row["convexity-quw"])

    frontier = np.array(frontier)
    suboptimal = np.array(suboptimal)
    natural = np.array(natural)
    points = np.concat((frontier, suboptimal, natural))
    check_frontier = np.concat((suboptimal, natural))
    check_natural = np.concat((frontier, suboptimal))
    return (
        frontier,
        suboptimal,
        natural,
        points,
        check_frontier,
        check_natural,
        convexities_qmw,
        convexities_quw,
    )


def get_neighbor_comparison(amount: int, model: pd.DataFrame):
    """
    Takes in a number of neighbors to compare to and the color model from the `.csv` file and displays information about neighboring encoders

    Args:
        amount (int): The number of neighbors to compare to.
        model (DataFrame): The DataFrame which can be directly loaded from the color model's minimized `.csv` file.
    """
    _, _, _, points, check_frontier, check_natural, convexities_qmw, convexities_quw = (
        convert_model(model)
    )

    comparison_qmw = []
    comparison_quw = []
    for i, p in enumerate(points):
        higher_than_qmw = 0
        higher_than_quw = 0
        if i < 1501:
            closest = get_closest(check_frontier, p, amount)
        elif i >= 2601:
            closest = get_closest(check_natural, p, amount)
        else:
            closest = get_closest(points, p, amount + 1)
        for c in closest:
            if i < 1501:
                c += 1501
            if c == i:
                continue
            if convexities_qmw[i] - convexities_qmw[c] > 0:
                higher_than_qmw += 1
            if convexities_qmw[i] - convexities_qmw[c] == 0:
                higher_than_qmw += 0.5
            if convexities_quw[i] - convexities_quw[c] > 0:
                higher_than_quw += 1
            if convexities_quw[i] - convexities_quw[c] == 0:
                higher_than_quw += 0.5
        comparison_qmw.append(higher_than_qmw / amount)
        comparison_quw.append(higher_than_quw / amount)

    print("Average comparison (q(m|w)):", sum(comparison_qmw) / len(comparison_qmw))
    print("Average natural language (q(m|w)):", sum(comparison_qmw[-110:]) / 110)
    print("Average optimal encoder (q(m|w)):", sum(comparison_qmw[:1501]) / 1501)
    print(
        "Average suboptimal encoder (q(m|w)):",
        sum(comparison_qmw[1501:-110]) / len(comparison_qmw[1501:-110]),
    )
    print()
    print("Average comparison (q(u|w)):", sum(comparison_quw) / len(comparison_quw))
    print("Average natural language (q(u|w)):", sum(comparison_quw[-110:]) / 110)
    print("Average optimal encoder (q(u|w)):", sum(comparison_quw[:1501]) / 1501)
    print(
        "Average suboptimal encoder (q(u|w)):",
        sum(comparison_quw[1501:-110]) / len(comparison_quw[1501:-110]),
    )


def check_difference_significance(amount: int, model: pd.DataFrame):
    """
    Takes in a number of neighbors to compare to and the color model from the `.csv` file and finds correlation between
    the quasi-convexity of a natural language's q(m|w) and q(u|w) distributions and that of their `amount` closest neighbors

    Args:
        amount (int): The number of neighbors to compare to.
        model (DataFrame): The DataFrame which can be directly loaded from the color model's minimized `.csv` file.
    """
    _, _, natural, _, _, check_natural, convexities_qmw, convexities_quw = (
        convert_model(model)
    )

    offset = len(check_natural)

    comparison_qmw = []
    comparison_quw = []

    for i, p in enumerate(natural):
        closest = get_closest(check_natural, p, amount)
        for c in closest:
            comparison_qmw.append(convexities_qmw[i + offset] - convexities_qmw[c])
            comparison_quw.append(convexities_quw[i + offset] - convexities_quw[c])

    print(ttest_1samp(comparison_qmw, 0))
    print(ttest_1samp(comparison_quw, 0))


if __name__ == "__main__":
    model = pd.read_csv("./colors/data/minimized/color_model.csv", header=0)

    check_difference_significance(10, model)
    get_neighbor_comparison(10, model)
