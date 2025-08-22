import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd

plt.rcParams["text.usetex"] = True

def gen_graph():
    plt.figure()

    model = pd.read_csv("./colors/data/minimized/color_model.csv", header=0)

    fig = plt.figure(figsize=(25, 15))

    # Create grid: last column (index COLS) is for colorbar
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.15])
    fig.subplots_adjust(wspace=0.4, hspace=0.4)

    axes = []
    qmw_ax = fig.add_subplot(gs[0, 0])
    scatter = fill_graph(
        qmw_ax, fig, model, True, add_colorbar=False, dotsize=50, quw=False
    )
    qmw_ax.set_title("Quasi-Convexity of $q(m|w)$", fontsize=50)
    qmw_ax.set_xlabel("Complexity, $I(M; W)$ (bits)", fontsize=40, labelpad=20)
    qmw_ax.set_ylabel("Accuracy, $I(W; U)$, (bits)", fontsize=40, labelpad=20)
    qmw_ax.tick_params(labelsize=30)
    axes.append(qmw_ax)

    quw_ax = fig.add_subplot(gs[0, 1])
    scatter = fill_graph(
        quw_ax, fig, model, True, add_colorbar=False, dotsize=50, quw=True
    )
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
        f"Accuracy vs Complexity Graphs with Quasi-Convexity\nof $q(m|w)$and $q(u|w)$ for Color Naming Encoders",
        fontsize=50,
    )
    plt.grid(False)
    plt.savefig(f"./colors/output/convexity/color.png")
    plt.close()


def fill_graph(
    ax: plt.Axes,
    fig: plt.Figure,
    model,
    convexity,
    add_colorbar=True,
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
    if add_colorbar:
        # Add colorbar
        cbar = fig.colorbar(scatter)
        cbar.set_label("Quasi-Convexity", rotation=270, labelpad=30, fontsize=20)
    return scatter

if __name__ == "__main__":
    gen_graph()
    