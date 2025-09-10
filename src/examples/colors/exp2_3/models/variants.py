from typing import Callable
from .utils import ModelSystem

from eff_conv.convexity import SimilaritySpace

import numpy as np


class CPUMSplit(ModelSystem):
    def __init__(self):
        pass

    def generate_meanings(self) -> np.ndarray:
        meanings = []
        for i in range(10):
            meaning = []
            for j in range(20):
                if j == i or j == i + 10:
                    meaning.append(0.41)
                else:
                    meaning.append(0.01)
            meanings.append(meaning)
        return np.array(meanings).T

    def meaning_priors(self) -> np.ndarray:
        return np.ones(10) / 10

    def space(self) -> SimilaritySpace:
        return SimilaritySpace(
            np.array([[i] for i in range(10)]), self.meaning_priors()
        )

    def name(self) -> str:
        return "cpum_split"


class CPDMSplit(ModelSystem):
    def __init__(self):
        pass

    def generate_meanings(self) -> np.ndarray:
        meanings = [[0.91, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]]
        for i in range(8):
            meanings.append([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        meanings.append([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.91])
        return np.array(meanings).T

    def meaning_priors(self) -> np.ndarray:
        return np.ones(10) / 10

    def space(self) -> SimilaritySpace:
        return SimilaritySpace(
            np.array([[i] for i in range(10)]), self.meaning_priors()
        )

    def name(self) -> str:
        return "cpdm_split"


# Convex Priors, Duplicate Meanings
class CPDMAdj(ModelSystem):
    def __init__(self, func: Callable):
        self.func = func

    def generate_meanings(self) -> np.ndarray:
        out = []
        for i in range(20):
            out.append(self.func(i // 2))
        return np.array(out).T

    def meaning_priors(self) -> np.ndarray:
        return np.ones(20) / 20.0

    def space(self) -> SimilaritySpace:
        return SimilaritySpace(
            np.array([[i] for i in range(20)]), self.meaning_priors()
        )

    def name(self) -> str:
        return "cpdm_adj"


# Convex Priors, Duplicate Meanings, convex on both ends
class CPDMConvex(ModelSystem):
    def __init__(self, func: Callable):
        self.func = func

    def generate_meanings(self) -> np.ndarray:
        out = []
        for i in range(-10, 11):
            out.append(self.func(abs(i)))
        return np.array(out).T

    def meaning_priors(self) -> np.ndarray:
        return np.array(
            [0.099, 0.099, 0.099, 0.099, 0.099, 0.099, 0.099, 0.099, 0.099, 0.099]
            + [0.01 / 11 for _ in range(11)]
        )

    def space(self) -> SimilaritySpace:
        return SimilaritySpace(
            np.array([[i] for i in range(-10, 11)]), self.meaning_priors()
        )

    def name(self) -> str:
        return "cpdm_convex"


class NPDMShift(ModelSystem):
    def __init__(self, func: Callable, suffix: str = ""):
        self.func = func
        self.suffix = suffix

    def generate_meanings(self) -> np.ndarray:
        out = []
        for i in range(-10, 11):
            out.append(self.func(abs(i)))
        return np.array(out).T

    def meaning_priors(self) -> np.ndarray:
        return np.array(
            [
                0.205,
                0.01,
                0.01,
                0.01,
                0.01,
                0.01,
                0.01,
                0.01,
                0.01,
                0.01,
                0.41,
                0.01,
                0.01,
                0.01,
                0.01,
                0.01,
                0.01,
                0.01,
                0.01,
                0.01,
                0.205,
            ]
        )

    def space(self) -> SimilaritySpace:
        return SimilaritySpace(
            np.array([[i] for i in range(-10, 11)]), self.meaning_priors()
        )

    def name(self) -> str:
        return f"npdm_shift{self.suffix}"


# Convex Priors, Duplicate Meanings
class NPDMAdj(ModelSystem):
    def __init__(self, func: Callable):
        self.func = func

    def generate_meanings(self) -> np.ndarray:
        out = []
        for i in range(18):
            out.append(self.func(i // 3))
        return np.array(out).T

    def meaning_priors(self) -> np.ndarray:
        return np.tile(np.array([0.495, 0.01, 0.495]) / 6, 6)

    def space(self) -> SimilaritySpace:
        return SimilaritySpace(
            np.array([[i] for i in range(18)]), self.meaning_priors()
        )

    def name(self) -> str:
        return "npdm_adj"
