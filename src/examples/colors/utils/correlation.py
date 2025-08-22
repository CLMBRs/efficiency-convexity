import pandas as pd


def csv_to_latex(file_name):
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
