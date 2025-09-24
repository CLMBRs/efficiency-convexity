from scipy.stats import pearsonr
import pandas as pd
import numpy as np


def calc_corr_to_tuple(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """
    Calculates the Person correlation coefficient and it's corresponding p values to 3 decimal places given 2 float tuples

    Args:
        x (np.ndarray): The independent variable values
        y (np.ndarray): The dependent variable values

    Returns:
        tuple[float, float]: The correlation coefficient (left) and the p value (right)
    """
    corr = pearsonr(x, y)
    c = round(corr.statistic, 3)
    p = round(corr.pvalue, 3)
    return (c, p)


def csv_to_latex(file_name: str) -> str:
    """
    Converts a correlation file into a custom-formatted LaTeX table and prints the LaTeX

    Args:
        file_name (str): The file name (without extension) of the correlation coefficients

    Returns:
        str: The formatting LaTeX table as a string
    """
    df = pd.read_csv(f"./colors/output/convexity/{file_name}.csv", header=0)
    body = []
    for _, row in df.iterrows():
        body.append(
            f"{row['env']} & {row['opt']} & {row['p opt']} & {row['I(M; W)']} & {row['p I(M; W)']} & {row['I(W; U)']} & {row['p I(W; U)']}\\\\"
        )
    body = "\n".join(body)
    return (
        """\\begin{table*}[t]
\\centering
\\begin{tabular}{lcccccc}
\\hline
\\multicolumn{7}{c}{TODO: TITLE}                   \\\\ \\hline
\\multicolumn{1}{c}{Environment} & Optimality & $p$ & $I(M; W)$ & $p$ & $I(W; U)$ & $p$ \\\\ \\hline
"""
        + body
        + """
\\end{tabular}
  \\caption{TODO: CAPTION}
  \\label{TODO: LABEL}
\\end{table*}"""
    )
