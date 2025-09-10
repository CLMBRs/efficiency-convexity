import statsmodels.formula.api as smf
import pandas as pd
import sys


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

    ols_from_model(model)
