import sys
from collections.abc import Iterable

import statsmodels.formula.api as smf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy import stats
from statsmodels.regression.linear_model import RegressionResults
from statsmodels.regression.mixed_linear_model import MixedLMResults
from statsmodels.stats.anova import anova_lm


def ols_from_model(
    model: pd.DataFrame,
    dependent_var: str = "convexity_qmw",
    independent_vars: Iterable[str] = ("optimality", "complexity", "accuracy"),
    interactions: bool = True,
) -> RegressionResults:
    """
    Prints the output for the OLS model describing `convexity_qmw ~ optimality + complexity * accuracy` and `convexity_quw ~ optimality + complexity * accuracy`
    for a given input DataFrame

    Args:
        model (DataFrame): The input DataFrame, the format should be one that is loaded from one of the `.csv` model files
    """

    results = smf.ols(
        f"{dependent_var} ~ {(' * ' if interactions else ' + ').join(independent_vars)}",
        data=model,
    ).fit()
    print(results.summary())
    return results

    """
    results = smf.ols(
        "convexity_quw ~ optimality + complexity * accuracy", data=model
    ).fit()
    print(results.summary())
    """


def mixed_lm_from_model(
    model: pd.DataFrame,
    dependent_var: str = "convexity_qmw",
    independent_vars: Iterable[str] = ("optimality", "complexity", "accuracy"),
    groups_col: str = "base_item_id",
    interactions: bool = True,
) -> MixedLMResults:
    """
    Prints the output for the mixed linear model describing `convexity_qmw ~ optimality + complexity * accuracy` with random intercepts for `base_item_id`
    for a given input DataFrame

    Args:
        model (DataFrame): The input DataFrame, the format should be one that is loaded from one of the `.csv` model files
    """

    lm = smf.mixedlm(
        f"{dependent_var} ~ {(' * ' if interactions else ' + ').join(independent_vars)}",
        data=model,
        groups=model[groups_col],  # random intercept
    )

    result = lm.fit(reml=False)
    print(result.summary())
    return result


def likelihood_ratio_test(
    mixed_model_result: MixedLMResults, ols_model_result: RegressionResults
) -> tuple[float, float, int]:
    """
    Performs a likelihood ratio test between a mixed linear model and an OLS model

    Args:
        mixed_model_result (MixedLMResults): The fitted mixed linear model result
        ols_model_result (RegressionResults): The fitted OLS model result

    Returns:
        tuple[float, float, int]: The likelihood ratio statistic, p-value, and degrees of freedom
    """

    lr_stat = 2 * (mixed_model_result.llf - ols_model_result.llf)
    p_value = stats.chi2.sf(lr_stat, df=1)  # 1 df for one random variance parameter
    return lr_stat, p_value, 1


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

    # first analysis: mixed vs. fixed effects for type + base_type
    # TODO: refactor this into its own method?

    mixed_model_results = mixed_lm_from_model(
        model,
        dependent_var="convexity_qmw",
        independent_vars=("type", "base_type"),
        interactions=False,
    )

    fixed_model_results = ols_from_model(
        model,
        dependent_var="convexity_qmw",
        independent_vars=("type", "base_type"),
        interactions=False,
    )

    lr_stat, p_value, df = likelihood_ratio_test(
        mixed_model_results, fixed_model_results
    )

    print(f"Likelihood Ratio Test: χ²(1) = {lr_stat:.3f}, p = {p_value:.4g}")

    print(f"Mixed model AIC: {mixed_model_results.aic:.2f}")
    print(f"OLS model AIC:   {fixed_model_results.aic:.2f}")
    print(f"Mixed model BIC: {mixed_model_results.bic:.2f}")
    print(f"OLS model BIC:   {fixed_model_results.bic:.2f}")

    big_model = smf.ols(
        "convexity_qmw ~ type + base_type +  optimality + complexity + accuracy",
        data=model,
    )
    big_model_result = big_model.fit()
    print(big_model_result.summary())

    interaction_model = smf.ols(
        "convexity_qmw ~ type + base_type + optimality * complexity * accuracy",
        data=model,
    )
    interaction_model_result = interaction_model.fit()
    print(interaction_model_result.summary())

    print(anova_lm(big_model_result, interaction_model_result))
    print(anova_lm(interaction_model_result))

    interaction_mixed_model = smf.mixedlm(
        "convexity_qmw ~ type + base_type + optimality * complexity * accuracy",
        data=model,
        groups=model["base_item_id"],
    )
    interaction_mixed_model_result = interaction_mixed_model.fit(reml=False)
    print(interaction_mixed_model_result.summary())
    lr_stat, p_value, df = likelihood_ratio_test(
        interaction_mixed_model_result, interaction_model_result
    )
    print(f"Likelihood Ratio Test: χ²(1) = {lr_stat:.3f}, p = {p_value:.4g}")

    medium_model = ols_from_model(
        model,
        dependent_var="convexity_qmw",
        independent_vars=("optimality", "complexity", "accuracy"),
        interactions=True,
    )
    print(anova_lm(medium_model, interaction_model_result))

    medium_mixed_model = mixed_lm_from_model(
        model,
        dependent_var="convexity_qmw",
        independent_vars=("optimality", "complexity", "accuracy"),
        interactions=True,
        groups_col="base_item_id",
    )
    lr_stat, p_value, df = likelihood_ratio_test(medium_mixed_model, medium_model)
    print(f"Likelihood Ratio Test: χ²(1) = {lr_stat:.3f}, p = {p_value:.4g}")

    print("\n\n AIC \t\t BIC \n")
    print(f"Big OLS Model: \t {big_model_result.aic:.2f} \t {big_model_result.bic:.2f}")
    print(f"Medium OLS Model: {medium_model.aic:.2f} \t {medium_model.bic:.2f}")
    print(
        f"Medium Mixed Model: {medium_mixed_model.aic:.2f} \t {medium_mixed_model.bic:.2f}"
    )
    print(
        f"Interaction OLS Model: {interaction_model_result.aic:.2f} \t {interaction_model_result.bic:.2f}"
    )
    print(
        f"Interaction Mixed Model: {interaction_mixed_model_result.aic:.2f} \t {interaction_mixed_model_result.bic:.2f}"
    )

    """
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
    plot_df["predicted_convexity"] = fixed_model_results.predict(exog=plot_df)

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
    """
