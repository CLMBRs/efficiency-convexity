from .utils import ModelSystem

from eff_conv.convexity import SimilaritySpace

import numpy as np


# Convex Priors, Unique Meanings
# Updated to fit paper 7/11
class CPUM(ModelSystem):
    def __init__(self, func, suffix=""):
        self.func = func
        self.suffix = suffix

    def generate_meanings(self):
        out = []
        for i in range(11):
            out.append(self.func(i))
        return np.array(out).T

    def meaning_priors(self):
        return np.ones(11) / 11.0

    def space(self):
        return SimilaritySpace(
            np.array([[i] for i in range(11)]), self.meaning_priors()
        )

    def name(self):
        return f"cpum{self.suffix}"


# Non-convex Priors, Unique Meanings
# Updated to fit paper 7/11
class NPUM(ModelSystem):
    def __init__(self, func, suffix=""):
        self.func = func
        self.suffix = suffix

    def generate_meanings(self):
        out = []
        for i in range(11):
            out.append(self.func(i))
        return np.array(out).T

    def meaning_priors(self):
        return np.array(
            [0.455, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.455]
        )

    def space(self):
        return SimilaritySpace(
            np.array([[i] for i in range(11)]), self.meaning_priors()
        )

    def name(self):
        return f"npum{self.suffix}"


# Convex Priors, Duplicate Meanings
# Updated to fit paper 7/11
class CPDM(ModelSystem):
    def __init__(self, func, suffix=""):
        self.func = func
        self.suffix = suffix

    def generate_meanings(self):
        out = []
        for i in range(-10, 11):
            out.append(self.func(abs(i)))
        return np.array(out).T

    def meaning_priors(self):
        return np.ones(21) / 21.0

    def space(self):
        return SimilaritySpace(
            np.array([[i] for i in range(-10, 11)]), self.meaning_priors()
        )

    def name(self):
        return f"cpdm{self.suffix}"


# Non-convex Priors, Duplicate Meanings
# Updated to fit paper 7/11
class NPDM(ModelSystem):
    def __init__(self, func, suffix=""):
        self.func = func
        self.suffix = suffix

    def generate_meanings(self):
        out = []
        for i in range(-10, 11):
            out.append(self.func(abs(i)))
        return np.array(out).T

    def meaning_priors(self):
        return np.array(
            [
                0.405,
                0.01,
                0.01,
                0.01,
                0.01,
                0.01,
                0.01,
                0.01,
                0.01,
                0.01,
                0.01,
                0.01,
                0.01,
                0.01,
                0.01,
                0.01,
                0.01,
                0.01,
                0.01,
                0.01,
                0.405,
            ]
        )

    def space(self):
        return SimilaritySpace(
            np.array([[i] for i in range(-10, 11)]), self.meaning_priors()
        )

    def name(self):
        return f"npdm{self.suffix}"
