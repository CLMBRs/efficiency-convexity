import statsmodels.formula.api as smf
import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt

from scipy import stats


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
        "convexity_qmw ~ optimality + type + base_type",
        data=model,
        groups=model["base_item_id"],  # random intercept for base_item_id
    )

    """
	lm_reduced = smf.mixedlm(
		"convexity_qmw ~ optimality + base_type",
		data=model,
		groups=model["base_item_id"],  # random intercept for base_item_id
	)
	"""

    result = lm.fit(reml=False)
    print(result.summary())

    fixed_model_results = smf.ols(
        "convexity_qmw ~ optimality + type + base_type", data=model
    ).fit()
    print(fixed_model_results.summary())

    # test statistic = 2 * (LL_mixed - LL_OLS)
    lr_stat = 2 * (result.llf - fixed_model_results.llf)
    p_value = stats.chi2.sf(lr_stat, df=1)  # 1 df for one random variance parameter

    print(f"Likelihood Ratio Test: χ²(1) = {lr_stat:.3f}, p = {p_value:.4g}")

    print(f"Mixed model AIC: {result.aic:.2f}")
    print(f"OLS model AIC:   {fixed_model_results.aic:.2f}")
    print(f"Mixed model BIC: {result.bic:.2f}")
    print(f"OLS model BIC:   {fixed_model_results.bic:.2f}")

    big_model = smf.ols(
        "convexity_qmw ~ type + base_type +  optimality + complexity + accuracy",
        data=model,
    )
    print(big_model.fit().summary())

    big_model = smf.ols(
        "convexity_qmw ~ type + base_type +  optimality * complexity * accuracy",
        data=model,
    )
    big_model_result = big_model.fit()
    print(big_model_result.summary())

    # --- Define ranges for your predictors ---
    opt_range = np.linspace(model["optimality"].min(), model["optimality"].max(), 100)
    comp_levels = np.linspace(
        model["complexity"].quantile(0.1), model["complexity"].quantile(0.9), 3
    )
    acc_levels = np.linspace(
        model["accuracy"].quantile(0.1), model["accuracy"].quantile(0.9), 3
    )

    # --- Helper: predict convexity from fitted model ---
    def predict_convexity(opt, comp, acc):
        """Use big_model_result.predict() for consistent term handling."""
        df = pd.DataFrame(
            {
                "optimality": opt,
                "complexity": comp,
                "accuracy": acc,
                # Include categorical predictors at reference levels:
                "type": "natural",  # replace with your actual reference level if needed
                "base_type": "natural",  # same here
            }
        )
        return big_model_result.predict(df)

    # --- Plot convexity vs. optimality at different levels ---
    plt.figure(figsize=(9, 6))
    for comp in comp_levels:
        for acc in acc_levels:
            df_pred = pd.DataFrame(
                {
                    "optimality": opt_range,
                    "complexity": comp,
                    "accuracy": acc,
                    "type": "natural",
                    "base_type": "natural",
                }
            )
            preds = big_model_result.predict(df_pred)
            label = f"Complexity={comp:.2f}, Accuracy={acc:.2f}"
            plt.plot(opt_range, preds, label=label)

    plt.xlabel("Optimality")
    plt.ylabel("Predicted Convexity (qmw)")
    plt.title(
        "Predicted convexity vs optimality at different complexity and accuracy levels"
    )
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.show()

    # --- 3D surface: Convexity as a function of Optimality × Complexity ---
    opt_grid, comp_grid = np.meshgrid(
        np.linspace(model["optimality"].min(), model["optimality"].max(), 60),
        np.linspace(model["complexity"].min(), model["complexity"].max(), 60),
    )
    acc_fixed = model["accuracy"].median()

    df_surface = pd.DataFrame(
        {
            "optimality": opt_grid.ravel(),
            "complexity": comp_grid.ravel(),
            "accuracy": acc_fixed,
            "type": "natural",
            "base_type": "natural",
        }
    )
    conv_grid = big_model_result.predict(df_surface).values.reshape(opt_grid.shape)

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(opt_grid, comp_grid, conv_grid, cmap="viridis", alpha=0.9)
    ax.set_xlabel("Optimality")
    ax.set_ylabel("Complexity")
    ax.set_zlabel("Predicted Convexity")
    ax.set_title(f"Interaction surface (Accuracy={acc_fixed:.2f})")
    plt.tight_layout()
    plt.show()

    df = model

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

    # Generate a grid of optimality values for prediction
    optimality_range = np.linspace(df["optimality"].min(), df["optimality"].max(), 100)

    # Define valid type/base_type combinations
    valid_combinations = [
        ("natural", "natural"),
        ("optimal", "optimal"),
        ("suboptimal", "natural"),
        ("suboptimal", "optimal"),
    ]

    # Build plot dataframe for predictions
    plot_df = pd.DataFrame(
        [
            {"optimality": o, "type": t, "base_type": b}
            for o in optimality_range
            for t, b in valid_combinations
        ]
    )

    # Predict using fixed effects only
    plot_df["predicted_convexity"] = result.predict(exog=plot_df)

    # Define colors for types
    type_colors = {"natural": "blue", "optimal": "green", "suboptimal": "red"}

    # Define line styles for base_type
    base_styles = {"natural": "-", "optimal": "--"}
    base_markers = {"natural": "o", "optimal": "s"}

    # Plot
    fig, ax = plt.subplots(figsize=(9, 6))

    # Plot predicted lines
    for t, b in valid_combinations:
        subset = plot_df[(plot_df["type"] == t) & (plot_df["base_type"] == b)]
        ax.plot(
            subset["optimality"],
            subset["predicted_convexity"],
            color=type_colors[t],
            linestyle=base_styles[b],
            label=f"Predicted: type={t}, base_type={b}",
        )

    # Overlay raw data
    for t in df["type"].unique():
        for b in df["base_type"].unique():
            subset = df[(df["type"] == t) & (df["base_type"] == b)]
            if len(subset) > 0:
                ax.scatter(
                    subset["optimality"],
                    subset["convexity_qmw"],
                    color=type_colors[t],
                    marker=base_markers[b],
                    alpha=0.5,
                    edgecolor="k",
                    s=30,
                    label=f"Raw: type={t}, base_type={b}",
                )

    ax.set_xlabel("Optimality")
    ax.set_ylabel("Convexity (qmw)")
    ax.set_title(
        "Predicted convexity vs optimality by type and base_type with raw data"
    )
    ax.legend(loc="best", fontsize=8, ncol=2)
    plt.tight_layout()
    plt.show()


# ols_from_model(model)
