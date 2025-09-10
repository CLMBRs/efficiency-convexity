from .utils import ModelSystem

from eff_conv.convexity import SimilaritySpace

import numpy as np


class ManhattanDistance(ModelSystem):
    def __init__(self, width: int, height: int):
        self.w = width
        self.h = height

    def generate_meanings(self) -> np.ndarray:
        meanings = []
        for y in range(self.h):
            for x in range(self.w):
                meaning = []
                for y1 in range(self.h):
                    for x1 in range(self.w):
                        meaning.append(float(abs(y - y1) + abs(x - x1)))
                meaning = np.array(meaning) + 1
                meaning /= np.sum(meaning)
                meanings.append(meaning)

        return np.array(meanings).T

    def meaning_priors(self) -> np.ndarray:
        return np.ones(self.w * self.h) / (self.w * self.h)

    def space(self) -> SimilaritySpace:
        points = []
        for y in range(self.h):
            for x in range(self.w):
                points.append([x, y])
        return SimilaritySpace(np.array(points), self.meaning_priors())

    def name(self) -> str:
        return f"manhattan_{self.w}_{self.h}"
