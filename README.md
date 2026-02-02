# Efficiency & Convexity

## Notes

This repository accompanies the paper "When Efficient Communication Explains Convexity." For code and data related to that paper, please navigate to `src/examples/colors`.

## Installing Efficiency & Convexity

1. Download or clone this repository and navigate to the root folder.

2. Install the IB Optimizer (We recommend doing this inside a virtual environment)

   `pip install -e .`

## Development, Examples, & Tests

### Setting Up the Environment

First, install [uv](https://docs.astral.sh/uv/getting-started/installation/). After `uv` is installed navigate to root of the project and execute:

```sh
uv sync
```

### Testing

Tests are located in `src/tests` and can be run using:

```sh
uv run pytest
```

### Examples

Examples can be found in the `src/examples` folder. Each example will have documentation of how to execute its scripts in their respective `README.md` files.
Note that oftentimes scripts from each example are to be executed in the `src/examples` folder, NOT the folder of the example itself.

## References

<details>
<summary>Links:</summary>

> Tishby, N., Fernando P., & Bialek, W. (2000). The information bottleneck method. The 37th annual Allerton Conference on Communication, Control, and Computing. pp. 368–377. https://doi.org/10.48550/arXiv.physics/0004057

> Skinner, L. (2025). Convexity is a Fundamental Feature of Efficient Semantic Compression in Probability Spaces. https://hdl.handle.net/1773/53008

> Zaslavsky, N., Kemp, C., Regier, T., & Tishby, N. (2018). Efficient compression in color naming and its evolution. Proceedings of the National Academy of Sciences, 115(31), 7937–7942. https://doi.org/10.1073/pnas.1800521115

</details>
