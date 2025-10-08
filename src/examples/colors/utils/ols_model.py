import statsmodels.formula.api as smf
import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt


def ols_from_model(model: pd.DataFrame):
    """
    Prints the output for the OLS model describing `convexity_qmw ~ optimality + complexity * accuracy` and `convexity_quw ~ optimality + complexity * accuracy`
    for a given input DataFrame

    Args:
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    model (DataFrame): The input DataFrame, the format should be one that is loaded from one of the `.csv` model files
    """
    model = model.rename(
        columns={"convexity-qmw": "convexity_qmw", "convexity-quw": "convexity_quw"}
    )

    results = smf.ols(
        "convexity_qmw ~ optimality + complexity * accuracy", data=model
    ).fit()
    print(results.summary())

    results = smf.ols(
        "convexity_quw ~ optimality + complexity * accuracy", data=model
    ).fit()
    print(results.summary())


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise ValueError("Invalid number of arguments, wanted 1")

    model = pd.read_csv(f"./colors/data/minimized/{sys.argv[1]}.csv", header=0)
    # Step 1: create a mapping from base_item_id → base type
    base_type_map = model.loc[
        model["item_id"] == model["base_item_id"], ["item_id", "type"]
    ].set_index("item_id")["type"]

    # Step 2: use that mapping to add a new column
    model["base_type"] = model["base_item_id"].map(base_type_map)

    model["type"] = model["type"].astype("category")
    model["base_type"] = model["base_type"].astype("category")

    model = model.rename(
        columns={"convexity-qmw": "convexity_qmw", "convexity-quw": "convexity_quw"}
    )

    lm = smf.mixedlm(
        "optimality ~ convexity_qmw + type + base_type",
        data=model,
        groups=model["base_item_id"],  # random intercept for base_item_id
    )

    result = lm.fit(reml=True)
    print(result.summary())

    df = model
    # Generate a grid of predictor values for plotting
    convexity_range = np.linspace(
        df["convexity_qmw"].min(), df["convexity_qmw"].max(), 100
    )

    # Create all combinations of type × base_type
    types = df["type"].unique()
    base_types = df["base_type"].unique()

    # Build a dataframe with all combinations
    # Only include valid type/base_type combinations
    valid_combinations = [
        ("natural", "natural"),
        ("optimal", "optimal"),
        ("suboptimal", "natural"),
        ("suboptimal", "optimal"),
    ]

    # Build plot dataframe
    plot_df = pd.DataFrame(
        [
            {"convexity_qmw": c, "type": t, "base_type": b}
            for c in convexity_range
            for t, b in valid_combinations
        ]
    )

    # Predict using fixed effects only
    plot_df["predicted_optimality"] = result.predict(exog=plot_df)

    # Define colors for types
    type_colors = {"natural": "blue", "optimal": "green", "suboptimal": "red"}

    # Define line styles for base_type
    base_styles = {"natural": "-", "optimal": "--"}

    # Plot
    fig, ax = plt.subplots(figsize=(9, 6))

    for t, b in valid_combinations:
        subset = plot_df[(plot_df["type"] == t) & (plot_df["base_type"] == b)]
        ax.plot(
            subset["convexity_qmw"],
            subset["predicted_optimality"],
            color=type_colors[t],
            linestyle=base_styles[b],
            label=f"type={t}, base_type={b}",
        )

    # Overlay raw data
    # Define marker styles for base_type in raw data
    base_markers = {"natural": "o", "optimal": "s"}
    for t, b in valid_combinations:
        subset = model[(model["type"] == t) & (model["base_type"] == b)]
        if len(subset) > 0:  # skip impossible combinations
            ax.scatter(
                subset["convexity_qmw"],
                subset["optimality"],
                color=type_colors[t],
                marker=base_markers[b],
                alpha=0.2,
                edgecolor="k",
                s=10,
                label=f"Raw: type={t}, base_type={b}",
            )

    ax.set_xlabel("Convexity (qmw)")
    ax.set_ylabel("Predicted optimality")
    ax.set_title("Predicted optimality vs convexity by type and base_type")
    ax.legend()
    plt.tight_layout()
    plt.show()


# ols_from_model(model)
