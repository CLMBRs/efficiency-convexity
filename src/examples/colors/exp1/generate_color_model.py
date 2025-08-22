from eff_conv.convexity import SimilaritySpace
from eff_conv.ib.language import IBLanguage
from eff_conv.ib.structure import IBStructure
from eff_conv.ib.optimization import run_deterministic_annealing
from eff_conv.ib.suboptimizer import shuffle_language

import pickle
import pandas as pd
import numpy as np
import multiprocessing as mp
import functools


def calculate_optimal(
    color_struct: IBStructure, betas: tuple[float, ...]
) -> list[IBLanguage]:
    optimized = run_deterministic_annealing(
        color_struct,
        betas,
        True,
        IBLanguage(structure, np.identity(color_struct.pum.shape[0])),
    )

    return [o[0] for o in optimized]


def generate_natural(lang_df: pd.DataFrame, struct: IBStructure) -> list[IBLanguage]:
    lang_terms = [{} for _ in range(int(lang_df["lang"].max()))]
    for _, row in lang_df.iterrows():
        if row["term"] in lang_terms[row["lang"] - 1]:
            lang_terms[row["lang"] - 1][row["term"]][row["chip"] - 1] += 1
        else:
            lang_terms[row["lang"] - 1][row["term"]] = [
                1 if i + 1 == row["chip"] else 0 for i in range(330)
            ]

    langs = []
    for terms in lang_terms:
        qwm = np.array(list(terms.values())).astype(float)
        qwm /= qwm.sum(axis=0)
        langs.append(IBLanguage(struct, qwm))

    return langs


def generate_suboptimal(langs: list[IBLanguage]) -> list[IBLanguage]:
    suboptimized = []

    for o in langs:
        for i in range(1, 21):
            for _ in range(3):
                suboptimized.append(shuffle_language(o, i / 20))

    return suboptimized


def calc_quw(sim_space: SimilaritySpace, lang: IBLanguage) -> float:
    return sim_space.language_convexity(lang, referents=True)


def calculate_convexities(
    optimal: list[IBLanguage],
    natural: list[IBLanguage],
    suboptimal: list[IBLanguage],
    sim_space: SimilaritySpace,
) -> tuple[
    tuple[list[float], list[float], list[float]],
    tuple[list[float], list[float], list[float]],
]:
    THREADS = 16

    optimal_convexity_quw = []
    suboptimal_convexity_quw = []
    natural_convexity_quw = []
    optimal_convexity = []
    suboptimal_convexity = []
    natural_convexity = []

    partial = functools.partial(calc_quw, sim_space)
    print("Calculating optimal convexity for q(u|w)", flush=True)
    with mp.Pool(processes=THREADS) as pool:
        if True:
            optimal_convexity_quw = pool.map(partial, optimal)
            with open("./colors/data/convexity/convexity_optimal_quw.pkl", "wb") as f:
                pickle.dump(optimal_convexity_quw, f)
        else:
            with open("./colors/data/convexity/convexity_optimal_quw.pkl", "rb") as f:
                optimal_convexity_quw = pickle.load(f)
        print("Calculating suboptimal convexity for q(u|w)", flush=True)
        if True:
            suboptimal_convexity_quw = pool.map(partial, suboptimal)
            with open(
                "./colors/data/convexity/convexity_suboptimal_quw.pkl", "wb"
            ) as f:
                pickle.dump(suboptimal_convexity_quw, f)
        else:
            with open(
                "./colors/data/convexity/convexity_suboptimal_quw.pkl", "rb"
            ) as f:
                suboptimal_convexity_quw = pickle.load(f)
        print("Calculating natural language convexity for q(u|w)", flush=True)
        if True:
            natural_convexity_quw = pool.map(partial, natural)
            with open("./colors/data/convexity/convexity_natural_quw.pkl", "wb") as f:
                pickle.dump(natural_convexity_quw, f)
        else:
            with open("./colors/data/convexity/convexity_natural_quw.pkl", "rb") as f:
                natural_convexity_quw = pickle.load(f)

        print("Calculating optimal convexity for q(m|w)", flush=True)
        if True:
            optimal_convexity = pool.map(sim_space.language_convexity, optimal)
            with open("./colors/data/convexity/convexity_optimal.pkl", "wb") as f:
                pickle.dump(optimal_convexity, f)
        else:
            with open("./colors/data/convexity/convexity_optimal.pkl", "rb") as f:
                optimal_convexity = pickle.load(f)
        print("Calculating suboptimal convexity for q(m|w)", flush=True)
        if True:
            suboptimal_convexity = pool.map(sim_space.language_convexity, suboptimal)
            with open("./colors/data/convexity/convexity_suboptimal.pkl", "wb") as f:
                pickle.dump(suboptimal_convexity, f)
        else:
            with open("./colors/data/convexity/convexity_suboptimal.pkl", "rb") as f:
                suboptimal_convexity = pickle.load(f)
        print("Calculating natural language convexity for q(m|w)", flush=True)
        if True:
            natural_convexity = pool.map(sim_space.language_convexity, natural)
            with open("./colors/data/convexity/convexity_natural.pkl", "wb") as f:
                pickle.dump(natural_convexity, f)
        else:
            with open("./colors/data/convexity/convexity_natural.pkl", "rb") as f:
                natural_convexity = pickle.load(f)

    return (
        (optimal_convexity, natural_convexity, suboptimal_convexity),
        (optimal_convexity_quw, natural_convexity_quw, suboptimal_convexity_quw),
    )


if __name__ == "__main__":
    GEN_OPTIMAL = True
    GEN_NATURAL = True
    GEN_SUBOPTIMAL = True

    with open("./colors/data/model.pkl", "rb") as f:
        optimal_model = pickle.load(f)

    print("Loaded color model", flush=True)

    structure = IBStructure(optimal_model["pU_M"].T, optimal_model["pM"].T[0])

    optimal_langs = []
    if GEN_OPTIMAL:
        optimal_langs = calculate_optimal(structure, optimal_model["betas"][::-1])
        with open("./colors/data/convexity/color_optimal.pkl", "wb") as f:
            pickle.dump(optimal_langs, f)
    else:
        with open("./colors/data/convexity/color_optimal.pkl", "rb") as f:
            optimal_langs = pickle.load(f)

    print("Calculated optimal languages", flush=True)

    natural_langs = []
    if GEN_NATURAL:
        natural_df = pd.read_csv(
            "./colors/data/WCS-Data-20110316/term.txt",
            sep="\t",
            names=["lang", "speaker", "chip", "term"],
        )
        natural_df["lang"] = natural_df["lang"].astype(int)
        natural_df["chip"] = natural_df["chip"].astype(int)
        natural_langs = generate_natural(natural_df, structure)
        with open("./colors/data/convexity/color_natural.pkl", "wb") as f:
            pickle.dump(natural_langs, f)
    else:
        with open("./colors/data/convexity/color_natural.pkl", "rb") as f:
            natural_langs = pickle.load(f)

    print("Created natural languages", flush=True)

    suboptimal_langs = []
    if GEN_SUBOPTIMAL:
        suboptimal_langs = generate_suboptimal(optimal_langs + natural_langs)
        with open("./colors/data/convexity/color_suboptimal.pkl", "wb") as f:
            pickle.dump(suboptimal_langs, f)
    else:
        with open("./colors/data/convexity/color_suboptimal.pkl", "rb") as f:
            suboptimal_langs = pickle.load(f)

    print("Created suboptimal languages", flush=True)

    cielab_df = pd.read_csv("./colors/data/cnum-vhcm-lab-new.txt", sep="\t", header=0)
    cielab_df.columns = ["chip_id", "V", "H", "C", "m_hue", "m_value", "L", "A", "B"]
    cielab_df.sort_values(by="chip_id", inplace=True)

    u_coords = cielab_df[["L", "A", "B"]].values
    sim_space = SimilaritySpace(
        np.array(u_coords).astype(float), point_prior=optimal_model["pM"].T[0]
    )

    convexity_qmw, convexity_quw = calculate_convexities(
        optimal_langs, natural_langs, suboptimal_langs, sim_space
    )
    convexity_dict = {
        "qmw": {
            "optimal": convexity_qmw[0],
            "natural": convexity_qmw[1],
            "suboptimal": convexity_qmw[2],
        },
        "quw": {
            "optimal": convexity_quw[0],
            "natural": convexity_quw[1],
            "suboptimal": convexity_quw[2],
        },
    }

    model = {
        "structure": structure,
        "optimal": optimal_langs,
        "natural": natural_langs,
        "suboptimal": suboptimal_langs,
        "convexity": convexity_dict,
    }

    print("Writing model to file", flush=True)

    with open("./colors/data/convexity/color_model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("Wrote model to file!", flush=True)
