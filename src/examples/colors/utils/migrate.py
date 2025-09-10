import pickle
import sys


def migrate_file(name: str):
    """
    Migrates a given old model file from Experiments 2 & 3 to a more organized format. This should not longer need to be used.

    Args:
        name (str): The name (without the `model-` prefix or the `.pkl` suffix) of the model file
    """
    with open(f"./colors/data/convexity/model-{name}.pkl", "rb") as f:
        model = pickle.load(f)

    new_model = {
        "optimal": model["optimized"],
        "suboptimal": model["suboptimal"],
        "space": model["space"],
        "convexity": {
            "qmw": {
                "optimal": model["optimized_convexity"],
                "suboptimal": model["suboptimal_convexity"],
            },
            "quw": {
                "optimal": model["optimized_convexity_quw"],
                "suboptimal": model["suboptimal_convexity_quw"],
            },
        },
    }

    with open(f"./colors/data/convexity/model-{name}.pkl", "wb") as f:
        pickle.dump(new_model, f)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("migrate requires a file name")
        exit()
    migrate_file(sys.argv[1])
