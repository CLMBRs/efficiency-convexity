from math import pi, sqrt
import numpy as np
import pickle

from eff_conv.convexity import SimilaritySpace
from eff_conv.ib.language import IBLanguage
from eff_conv.ib.optimization import run_deterministic_annealing
from eff_conv.ib.structure import IBStructure
from eff_conv.ib.suboptimizer import shuffle_language

from .models.utils import ModelSystem
from .models.base import CPUM, CPDM, NPDM, NPUM
from .models.variants import (
    CPDMAdj,
    CPDMConvex,
    CPDMSplit,
    CPUMSplit,
    NPDMAdj,
    NPDMShift,
)
from .models.special import ManhattanDistance


def generate_model(system: ModelSystem, referent_space: np.ndarray, add_convexity=True):
    with open("./colors/data/model.pkl", "rb") as f:
        betas = pickle.load(f)["betas"][::-1]

    structure = IBStructure(system.generate_meanings(), system.meaning_priors())

    optimized = run_deterministic_annealing(
        structure,
        betas,
        True,
        IBLanguage(structure, np.identity(len(structure.meanings_prior))),
    )

    suboptimized = []

    for o, _ in optimized:
        for i in range(1, 11):
            suboptimized.append(shuffle_language(o, i / 10))

    optimal_convexity_quw = []
    suboptimal_convexity_quw = []
    space_quw = SimilaritySpace(referent_space, structure.referents_prior)

    if add_convexity:
        print("Calculating optimal convexity for q(u|w)")
        for o in optimized:
            optimal_convexity_quw.append(
                space_quw.language_convexity(o[0], referents=True)
            )
        print("Calculating suboptimal convexity for q(u|w)")
        for s in suboptimized:
            suboptimal_convexity_quw.append(
                space_quw.language_convexity(s, referents=True)
            )

    optimal_convexity = []
    suboptimal_convexity = []
    space = system.space()

    if add_convexity:
        print("Calculating optimal convexity for q(m|w)")
        for o in optimized:
            optimal_convexity.append(space.language_convexity(o[0]))
        print("Calculating suboptimal convexity for q(m|w)")
        for s in suboptimized:
            suboptimal_convexity.append(space.language_convexity(s))

    model = {
        "optimal": optimized,
        "suboptimal": suboptimized,
        "space": system.space(),
        "convexity": {
            "qmw": {
                "optimal": optimal_convexity,
                "suboptimal": suboptimal_convexity,
            },
            "quw": {
                "optimal": optimal_convexity_quw,
                "suboptimal": suboptimal_convexity_quw,
            },
        },
    }

    print(f"Writing {system.name()} to file")
    with open(f"./colors/data/convexity/model-{system.name()}.pkl", "wb") as f:
        pickle.dump(model, f)


SD = 1.5


def reg_dist(i):
    meaning = np.array(list(range(0, 11)))
    meaning = np.exp(-0.5 * ((meaning - i) / SD) ** 2) / (SD * sqrt(4 * pi))
    return meaning / np.sum(meaning)


def dual_dist(i):
    meaning = np.array(list(range(-10, 11)))
    meaning = (
        np.exp(-0.5 * ((meaning - i) / SD) ** 2)
        + np.exp(-0.5 * ((meaning + i) / SD) ** 2)
    ) / (SD * sqrt(8 * pi))
    return meaning / np.sum(meaning)


if __name__ == "__main__":
    generate_model(CPUM(reg_dist), np.array([[i] for i in range(0, 11)]))
    generate_model(CPDM(reg_dist), np.array([[i] for i in range(0, 11)]))
    generate_model(NPUM(reg_dist), np.array([[i] for i in range(0, 11)]))
    generate_model(NPDM(reg_dist), np.array([[i] for i in range(0, 11)]))
    generate_model(
        CPUM(dual_dist, suffix="_dual"), np.array([[i] for i in range(-10, 11)])
    )
    generate_model(
        CPDM(dual_dist, suffix="_dual"), np.array([[i] for i in range(-10, 11)])
    )
    generate_model(
        NPUM(dual_dist, suffix="_dual"), np.array([[i] for i in range(-10, 11)])
    )
    generate_model(
        NPDM(dual_dist, suffix="_dual"), np.array([[i] for i in range(-10, 11)])
    )
    generate_model(CPDMConvex(reg_dist), np.array([[i] for i in range(0, 11)]))
    generate_model(CPDMAdj(reg_dist), np.array([[i] for i in range(0, 11)]))
    generate_model(NPDMAdj(reg_dist), np.array([[i] for i in range(0, 11)]))
    points = []
    for y in range(5):
        for x in range(5):
            points.append([x, y])
    generate_model(ManhattanDistance(5, 5), np.array(points))
    generate_model(NPDMShift(reg_dist), np.array([[i] for i in range(0, 11)]))
    generate_model(CPUMSplit(), np.array([[i] for i in range(20)]))
    generate_model(CPDMSplit(), np.array([[i] for i in range(10)]))
