import pandas as pd
import pickle
import os


def minimize_model(name: str):
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
    }

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

    for i, s in enumerate(model["suboptimal"]):
        if artificial:
            df_data["beta"].append(-1)
        df_data["accuracy"].append(s.iwu)
        df_data["complexity"].append(s.complexity)
        df_data["convexity-qmw"].append(model["convexity"]["qmw"]["suboptimal"][i])
        df_data["convexity-quw"].append(model["convexity"]["quw"]["suboptimal"][i])
        df_data["type"].append("suboptimal")

    if not artificial:
        for i, n in enumerate(model["natural"]):
            df_data["accuracy"].append(n.iwu)
            df_data["complexity"].append(n.complexity)
            df_data["convexity-qmw"].append(model["convexity"]["qmw"]["natural"][i])
            df_data["convexity-quw"].append(model["convexity"]["quw"]["natural"][i])
            df_data["type"].append("natural")

        with open(f"./colors/data/model.pkl", "rb") as f:
            optimal_model = pickle.load(f)
        df_data["beta"] = list(optimal_model["betas"][::-1])
        for i in range(len(df_data["beta"]), len(df_data["accuracy"])):
            df_data["beta"].append(-1)

    df = pd.DataFrame(data=df_data)
    df.to_csv(f"./colors/data/minimized/{name}.csv")


if __name__ == "__main__":
    dir = "./colors/data/convexity"
    for entry in os.listdir(dir):
        full_path = os.path.join(dir, entry)
        if os.path.isfile(full_path) and entry[-4:] == ".pkl":
            print("Minimizing", entry)
            minimize_model(entry[:-4])
