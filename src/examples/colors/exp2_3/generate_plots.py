import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import math

plt.rcParams["text.usetex"] = True


def tiled_graph(names, titles, quw=False):
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
    model,
    dotsize=5,
    quw=False,
):
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
