from matplotlib.collections import PathCollection
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd


def gen_graph():
    """
    Generates and saves the graph for the color plot, which displays the accuracy against the complexity of the encoders, with natural
    language encoders circled and the dots colored based on quasi-convexity.
    """
    plt.figure()

    model = pd.read_csv("./colors/data/minimized/color_model.csv", header=0)

    fig = plt.figure(figsize=(25, 15))

    # Create grid: last column (index COLS) is for colorbar
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.15])
    fig.subplots_adjust(wspace=0.4, hspace=0.4)

    axes = []
    qmw_ax = fig.add_subplot(gs[0, 0])
    scatter = fill_graph(qmw_ax, model, dotsize=50, quw=False)
    qmw_ax.set_title("Quasi-Convexity of $q(m|w)$", fontsize=50)
    qmw_ax.set_xlabel("Complexity, $I(M; W)$ (bits)", fontsize=40, labelpad=20)
    qmw_ax.set_ylabel("Accuracy, $I(W; U)$, (bits)", fontsize=40, labelpad=20)
    qmw_ax.tick_params(labelsize=30)
    axes.append(qmw_ax)

    quw_ax = fig.add_subplot(gs[0, 1])
    scatter = fill_graph(quw_ax, model, dotsize=50, quw=True)
    quw_ax.set_title("Quasi-Convexity of $q(u|w)$", fontsize=50)
    quw_ax.set_xlabel("Complexity, $I(M; W)$ (bits)", fontsize=40, labelpad=20)
    quw_ax.set_ylabel("Accuracy, $I(W; U)$, (bits)", fontsize=40, labelpad=20)
    quw_ax.tick_params(labelsize=30)
    axes.append(quw_ax)

    # Add a single colorbar in the last column, spanning all rows
    cax = fig.add_subplot(gs[:, -1])
    cbar = fig.colorbar(scatter, cax=cax)
    cbar.ax.tick_params(labelsize=30)
    cbar.set_label("Quasi-Convexity", rotation=90, fontsize=50, labelpad=30)
    cbar.ax.yaxis.set_label_position("left")

    fig.subplots_adjust(top=0.9, bottom=0.06, left=0.06, right=0.94)

    # Label axes and show plot
    fig.suptitle(
        f"Accuracy vs Complexity Graphs with Quasi-Convexity\nof $q(m|w)$ and $q(u|w)$ for Color Naming Encoders",
        fontsize=50,
    )
    plt.grid(False)
    plt.savefig(f"./colors/output/convexity/color.png")
    plt.close()


def fill_graph(
    ax: plt.Axes,
    model: pd.DataFrame,
    dotsize: int = 5,
    quw: bool = False,
) -> PathCollection:
    """
    Generates a scatter plot for the model plotting the accuracy against the complexity, with coloring based on the convexity

    Args:
        ax (plt.Axes): The axes for the subplot which will be filled
        model (DataFrame): The DataFrame for the model which can be loaded directly from the `.csv` file
        dotsize (int): The size of the dots for the scatter plot
        quw (bool): Whether the quasi-convexity of the q(u|w) or the q(m|w) distribution is being used

    Returns:
        PathCollection: The scatter plot
    """
    ax.set_box_aspect(1)

    convexity = model[f"convexity-q{'u' if quw else 'm'}w"].tolist()

    scatter = ax.scatter(
        model["complexity"].tolist(),
        model["accuracy"].tolist(),
        c=convexity,
        cmap="viridis",
        vmin=0,
        vmax=1,
        s=dotsize,
    )

    natural = model[model["type"] == "natural"]

    ax.scatter(
        natural["complexity"].tolist(),
        natural["accuracy"].tolist(),
        c=natural[f"convexity-q{'u' if quw else 'm'}w"],
        cmap="viridis",
        vmin=0,
        vmax=1,
        s=dotsize,
        edgecolors="grey",
        linewidths=2,
    )

    return scatter


if __name__ == "__main__":
    plt.rcParams["text.usetex"] = True
    gen_graph()
