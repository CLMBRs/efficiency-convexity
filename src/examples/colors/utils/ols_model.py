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
    Prints the output for the mixed linear model describing `convexity_qmw ~ optimality * complexity * accuracy` with random intercepts for `base_item_id`
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

    # Rename columns for easier access
    model = model.rename(
        columns={"convexity-qmw": "convexity_qmw", "convexity-quw": "convexity_quw"}
    )

    # first analysis: mixed vs. fixed effects for type + base_type
    # TODO: refactor these into methods?

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

    print(f"Likelihood Ratio Test: χ²(1) = {lr_stat:.3f}, p = {p_value:.4g}\n\n")

    # second analysis: full model with all predictors and interactions
    big_model = smf.ols(
        "convexity_qmw ~ type + base_type + optimality + complexity + accuracy",
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
