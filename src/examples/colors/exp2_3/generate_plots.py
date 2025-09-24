from matplotlib.collections import PathCollection
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import math


def tiled_graph(names: list[str], titles: list[str], quw: bool = False):
    """
    Generates a tiled graph with the names and respective titles displaying all of the listed systems

    Args:
        names (list[str]): The list of the systems to place into the graph
        titles (list[str]): The titles to associate with each of the systems
        quw (bool): Whether to use the quasi-convexity of the q(u|w) or q(m|w) distributions
    """
    plt.figure()

    COLS = 4
    ROWS = math.ceil(len(names) / COLS)

    fig = plt.figure(figsize=(37, 27))

    # Create grid: last column (index COLS) is for colorbar
    gs = gridspec.GridSpec(ROWS, COLS + 1, width_ratios=[1] * COLS + [0.15])
    fig.subplots_adjust(wspace=0.4, hspace=0.4)

    axes = []
    for i, name in enumerate(names):
        row = i // COLS
        col = i % COLS
        ax = fig.add_subplot(gs[row, col])
        model = pd.read_csv(f"./colors/data/minimized/model-{name}.csv", header=0)
        scatter = fill_graph(ax, model, dotsize=10, quw=quw)
        ax.set_title(f"\\textsc{'{' + titles[i] + '}'}", fontsize=50)
        ax.set_xlabel("Complexity, $I(M; W)$ (bits)", fontsize=30, labelpad=20)
        ax.set_ylabel("Accuracy, $I(W; U)$, (bits)", fontsize=30, labelpad=20)
        ax.tick_params(labelsize=30)
        axes.append(ax)

    # Remove unused subplots (if any)
    for i in range(len(names), ROWS * COLS):
        fig.add_subplot(gs[i // COLS, i % COLS]).remove()

    # Add a single colorbar in the last column, spanning all rows
    cax = fig.add_subplot(gs[:, -1])
    cbar = fig.colorbar(scatter, cax=cax)
    cbar.ax.tick_params(labelsize=50)
    cbar.set_label("Quasi-Convexity", rotation=90, fontsize=70, labelpad=30)
    cbar.ax.yaxis.set_label_position("left")

    fig.subplots_adjust(top=0.9, bottom=0.06, left=0.04, right=0.96)

    # Label axes and show plot
    fig.suptitle(
        f"Accuracy vs Complexity Graphs with Quasi-Convexity of ${'q(u|w)' if quw else 'q(m|w)'}$ for Various IB Environments",
        fontsize=70,
    )
    plt.grid(False)
    plt.savefig(f"./colors/output/convexity/tile{'_quw' if quw else '_qmw'}.png")
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

    return scatter


if __name__ == "__main__":
    plt.rcParams["text.usetex"] = True

    tiled_graph(
        [
            "cpum",
            "npum",
            "cpdm",
            "npdm",
            "cpum_dual",
            "npum_dual",
            "cpdm_dual",
            "npdm_dual",
            "cpdm_convex",
            "cpdm_adj",
            "npdm_adj",
            "manhattan_5_5",
        ],
        [
            "CPUM",
            "NPUM",
            "CPDM",
            "NPDM",
            "CPUM-Dual",
            "NPUM-Dual",
            "CPDM-Dual",
            "NPDM-Dual",
            "CPDM-Convex",
            "CPDM-Adj",
            "NPDM-Adj",
            "Manhattan-5-5",
        ],
    )

    tiled_graph(
        [
            "cpum",
            "npum",
            "cpdm",
            "npdm",
            "cpum_dual",
            "npum_dual",
            "cpdm_dual",
            "npdm_dual",
            "npdm_shift",
            "cpum_split",
            "cpdm_split",
            "manhattan_5_5",
        ],
        [
            "CPUM",
            "NPUM",
            "CPDM",
            "NPDM",
            "CPUM-Dual",
            "NPUM-Dual",
            "CPDM-Dual",
            "NPDM-Dual",
            "NPDM-Shift",
            "CPUM-Split",
            "CPDM-Split",
            "Manhattan-5-5",
        ],
        quw=True,
    )
