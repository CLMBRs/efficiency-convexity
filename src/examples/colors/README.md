## Colors

Based on the paper [TODO]

## Setup
After setting up the `eff_conv` environment, install the proper prerequisites:
```sh
python -m pip install -r requirements.txt
```

## How To Run

### Installing The Data

Please install the `model.pkl` following the direction from [here](https://github.com/nogazs/ib-color-naming/tree/master). Place the `model.pkl` file in the `data` folder

Please additionally install the `WCS-Data-20110316.zip` file from the World Color Survey data from [here](https://linguistics.berkeley.edu/wcs/data.html). Make sure the unzipped folder is in `data`. Make sure to also install the mappings (`cnum-vhcm-lab-new.txt`) and place them in the `data` folder as well.

### Creating the WCS and CIELab visualizations

TODO: The stimulus grid needs to be updated

In the `examples` directory, run the following commands to get CIELab visualization (Figure 2 in the paper):
```sh
python -m colors.visualization.cloud color_reg
python -m colors.visualization.cloud color_rot 20
python -m colors.visualization.cloud color_diff 20 diff
```

This will create the files `color_reg.png`, `color_rot.png`, and `color_diff.png` in the `output/color` folder.


### Minimizing Data

The raw model files for each experiement are very large (ranging from 250 MB for artifical models to over 14 GB for the color model). In order to distribute the data and for ease of iteration, we provide minimized `.csv` files containing the relevant data in the GitHub repository. We also include the raw model files we generated [here](https://osf.io/x3dkz/).

If you wish to run this experiment with your own data, after doing the data generation for experiment 1 and 2 you will need to generate the `.csv` files using the following command:

```sh
python -m colors.utils.minimize_models
```

### Experiment 1

#### Data Generation

To generate the color model run the following command in the `examples` directory:
```sh
python -m colors.exp1.generate_color_model
```

#### Data Analysis

Please ensure you have the minimized `.csv` files before doing analysis.

You can generate the plots and correlations using the respective commands:
```sh
# Plots
python -m colors.exp1.generate_color_plot
python -m colors.exp1.gen_whisker_plots
# Correlations
python -m colors.exp1.generate_color_correlation
python -m colors.utils.ols_model color_model
```

TODO: ADD OTHER CORRELATION MODEL AND OTHER ANALYSIS

### Experiments 2 & 3

#### Data Generation

To generate the test models for experiments 2 and 3, you can run the followign command in the `examples` directory:
```sh
python -m colors.exp2_3.generate_models
```

### Data Analysis

Please ensure you have the minimized `.csv` files before doing analysis.

After this you can generate the plots and correlations using the respective commands:
```sh
python -m colors.exp2_3.generate_plots
python -m colors.exp2_3.calculate_correlation
```

TODO: ADD OTHER CORRELATION MODEL AND OTHER ANALYSIS

## Additional Documentation
- `exp1.check_diff`
  - Experimental: Used to get the data regarding the differences with nearby neighbors
- `exp1.gen_whisker_plots`
  - The script to generate the whisker plots in the paper. The output will be in `output/convexity/whisker.png`
- `exp1.generate_color_correlation`
  - The script to generate the color correlation file and table. The correlation file will be in `output/convexity/color_coefficients.csv`
- `exp1.generate_color_model`
  - The script to generate the color model. The model file will be in `data/convexity/color_model.pkl`. This script will take a few days to run.
- `exp1.generate_color_plot`
  - The script to generate the color plot. The model file will be in `output/output/color.png`.
- `exp2_3.models.*`
  - The scaffolding for the various artifical models showcased in the paper.
- `exp2_3.calculate_correlation`
  - The script to generate the convexity correlation files and tables. The correlation files will be in `output/convexity/qmw_coefficients.csv` and `output/convexity/quw_coefficients.csv`
- `exp2_3.generate_models`
  - The script to generate the artificial model files. The model files will be in `data/convexity/model-*.pkl`
- `exp2_3.generate_plots`
  - The script to generate the artificial model plots. The plots will be in `output/convexity/tile_qmw.png` and `output/convexity/tile_quw.png`
- `utils.correlation`
  - This contains a function to convert the correlation tables into a formatted LaTeX table
- `utils.migrate`
  - This was a script to fix discrepencies between an old format of models and a more organized format
- `utils.minimize_models`
  - This minimizes the models in the `data/convexity/` folder into `.csv` files and outputs them in `data/minimized/`
- `utils.ols_model`
  - Takes in an argument, that being the name of a minimized `.csv` file without the extension. Creates an OLS model for the given model.
- `visualization.cloud`
  - Takes in 1 - 3 arguemnts, the first being the name of the output file and the second (optional) being a rotation value, the final can argument is `diff` and toggles whether or not to show difference point cloud or the regular point cloud.
- `visualization.gen_wcs_grid`
  - Generates a very simplistic WCS stimulus grid. To change the rotation a constant must be manually edited in the file.

## Notes

If you found this useful please cite the following paper:
[TODO]