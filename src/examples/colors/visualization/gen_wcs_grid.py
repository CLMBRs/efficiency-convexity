from skimage import color
from PIL import Image, ImageDraw
import numpy as np
import pandas as pd

cielab_df = pd.read_csv("./colors/data/cnum-vhcm-lab-new.txt", sep="\t", header=0)
cielab_df.columns = ["chip_id", "V", "H", "C", "m_hue", "m_value", "L", "A", "B"]
# cielab_df.sort_values(by="chip_id", inplace=True)
raw_points = cielab_df[["L", "A", "B"]].values
points = np.array(raw_points)

ROTATION = 0
if ROTATION > 0:
    colors = []
    for _, row in cielab_df.iterrows():
        if int(row["H"]) == 0:
            colors.append(row[["L", "A", "B"]].values)
            continue
        h_val = (int(row["H"]) + ROTATION - 1) % 40 + 1
        new_color = cielab_df.loc[
            (cielab_df["H"] == h_val) & (cielab_df["V"] == row["V"])
        ].iloc[0]
        colors.append(new_color[["L", "A", "B"]].values)
    colors = np.array(colors).astype(np.float64)
    colors = color.lab2rgb(colors)
else:
    colors = color.lab2rgb(points)

img = Image.new("RGB", (1850, 455), color="white")
draw = ImageDraw.Draw(img)

for i, c in enumerate(colors):
    color = (int(c[0] * 255), int(c[1] * 255), int(c[2] * 255))
    if i == 0:
        draw.rectangle([(5, 5), (45, 45)], fill=color)
        continue
    if i == len(colors) - 1:
        draw.rectangle([(5, 410), (45, 450)], fill=color)
        continue
    x_coord = (i - 1) % 41
    y_coord = (i - 1) // 41 + 1
    draw.rectangle(
        [
            (40 * x_coord + 5 * (x_coord + 1), 40 * y_coord + 5 * (y_coord + 1)),
            (
                40 * (x_coord + 1) + 5 * (x_coord + 1),
                40 * (y_coord + 1) + 5 * (y_coord + 1),
            ),
        ],
        fill=color,
    )

img.save("./colors/output/color/grid.png")
