import matplotlib.pyplot as plt
import pandas as pd


def generate_whisker_plot():
    """
    Generates a whisker plot showing the convexities for optimal, suboptimal, and natural language encoders from the color
    naming data.
    """
    model = pd.read_csv("./colors/data/minimized/color_model.csv", header=0)

    convexities = {
        "qmw": {
            "optimal": [],
            "suboptimal": [],
            "natural": [],
        },
        "quw": {
            "optimal": [],
            "suboptimal": [],
            "natural": [],
        },
    }

    for _, row in model.iterrows():
        convexities["qmw"][row["type"]].append(row["convexity-qmw"])
        convexities["quw"][row["type"]].append(row["convexity-quw"])

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(8, 4), sharex=True, dpi=600)

    flier_props = {
        "marker": "o",
        "markersize": 2,
        "markerfacecolor": "black",
        "markeredgecolor": "black",
    }
    # Plot each horizontal boxplot
    for i, n in enumerate(["optimal", "suboptimal", "natural"]):
        ax = axes[i, 0]
        ax.set_title(f"{n[0].upper()}{n[1:]} encoders")
        ax.boxplot(
            convexities["qmw"][n],
            orientation="horizontal",
            widths=0.75,
            flierprops=flier_props,
        )
        ax.set_yticks([])
        ax.set_xlim(0, 1)

    for i, n in enumerate(["optimal", "suboptimal", "natural"]):
        ax = axes[i, 1]
        ax.set_title(f"{n[0].upper()}{n[1:]} encoders")
        ax.boxplot(
            convexities["quw"][n],
            orientation="horizontal",
            widths=0.75,
            flierprops=flier_props,
        )
        ax.set_yticks([])
        ax.set_xlim(0, 1)

    # Add labels
    axes[2, 0].set_xlabel("Quasi-Convexity of $q(m|w)$")
    axes[2, 1].set_xlabel("Quasi-Convexity of $q(u|w)$")

    plt.tight_layout()
    plt.savefig(f"./colors/output/convexity/whisker.png")


if __name__ == "__main__":
    plt.rcParams["text.usetex"] = True

    generate_whisker_plot()
