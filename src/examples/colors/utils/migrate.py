import pickle
import sys


def migrate_file(name):
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
