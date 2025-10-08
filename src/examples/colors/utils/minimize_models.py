import pandas as pd
import numpy as np
import pickle
import os


def find_frontier_optimality(frontier: np.ndarray, point: np.ndarray) -> float:
    """
    Calculates the largest negative distance (smallest absolute value) of a point to a group of points (frontier)

    Args:
        frontier (np.ndarray): A list of 2-dimensional points. This is the points which the input point is checked against.
        point (np.ndarray): A 2-dimensional point. This is the point which is being checked.

    Returns:
        float: the negative distance from the input point to the closest point in frontier.
    """
    return -np.min(np.linalg.norm(frontier - point, axis=1))


def minimize_model(name: str):
    """
    Converts a model file consisting of various languages to a `.csv` file which has a row per encoder contains the following columns:
    accuracy: The accuracy of the encoder's q(w|m) distribution. This is the same as IBLanguage.iwu
    complexity: The complexity of the encoder's q(w|m) distribution. This is the same as IBLanguage.complexity
    convexity-qmw: The quasi-convexity of the encoder's q(m|w) distribution.
    convexity-quw: The quasi-convexity of the encoder's q(u|w) distribution.
    type: The type of encoder (optimal, suboptimal, or natural).
    beta: The beta value for optimal encoders. If the encoder is not optimal this value is -1.
    optimality: The negative distance to the nearest optimal encoder.

    Args:
        name: the file name (without extension) to convert.
    """
    with open(f"./colors/data/convexity/{name}.pkl", "rb") as f:
        model = pickle.load(f)

    artificial = "natural" not in model

    df_data = {
        "accuracy": [],
        "complexity": [],
        "convexity-qmw": [],
        "convexity-quw": [],
        "type": [],
        "beta": [],
        "optimality": [],
        "base_item_id": []
    }

    frontier = []

    for i, o in enumerate(model["optimal"]):
        if artificial:
            lang, beta = o
            df_data["beta"].append(beta)
        else:
            lang = o
        df_data["accuracy"].append(lang.iwu)
        df_data["complexity"].append(lang.complexity)
        df_data["convexity-qmw"].append(model["convexity"]["qmw"]["optimal"][i])
        df_data["convexity-quw"].append(model["convexity"]["quw"]["optimal"][i])
        df_data["type"].append("optimal")
        df_data["optimality"].append(0)
        df_data["base_item_id"].append(i)
        frontier.append([lang.complexity, lang.iwu])

    frontier = np.array(frontier)

    if not artificial:
        offset = len(model["optimal"])
        for i, n in enumerate(model["natural"]):
            df_data["accuracy"].append(n.iwu)
            df_data["complexity"].append(n.complexity)
            df_data["convexity-qmw"].append(model["convexity"]["qmw"]["natural"][i])
            df_data["convexity-quw"].append(model["convexity"]["quw"]["natural"][i])
            df_data["type"].append("natural")
            df_data["optimality"].append(
                find_frontier_optimality(frontier, np.array([n.complexity, n.iwu]))
            )
            df_data["base_item_id"].append(i + offset)

        with open(f"./colors/data/model.pkl", "rb") as f:
            optimal_model = pickle.load(f)
        df_data["beta"] = list(optimal_model["betas"][::-1])
        for i in range(len(df_data["beta"]), len(df_data["accuracy"])):
            df_data["beta"].append(-1)

    for i, s in enumerate(model["suboptimal"]):
        df_data["beta"].append(-1)
        df_data["accuracy"].append(s.iwu)
        df_data["complexity"].append(s.complexity)
        df_data["convexity-qmw"].append(model["convexity"]["qmw"]["suboptimal"][i])
        df_data["convexity-quw"].append(model["convexity"]["quw"]["suboptimal"][i])
        df_data["type"].append("suboptimal")
        df_data["optimality"].append(
            find_frontier_optimality(frontier, np.array([s.complexity, s.iwu]))
        )
        df_data["base_item_id"].append(i//10)

    df = pd.DataFrame(data=df_data)
    df.index.name = "item_id"
    df.to_csv(f"./colors/data/minimized/{name}.csv")


if __name__ == "__main__":
    dir = "./colors/data/convexity"
    for entry in os.listdir(dir):
        full_path = os.path.join(dir, entry)
        if os.path.isfile(full_path) and entry[-4:] == ".pkl":
            print("Minimizing", entry)
            minimize_model(entry[:-4])
