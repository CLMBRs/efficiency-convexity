from eff_conv.language import IBLanguage

from scipy.spatial import ConvexHull

import numpy as np


class SimilaritySpace:
    sim_space: np.ndarray
    point_prior: np.ndarray

    def __init__(self, sim_space: np.ndarray, point_prior: np.ndarray = None):
        if len(sim_space.shape) != 2:
            raise ValueError("Similarity space input must be a 2d matrix")
        self.sim_space = sim_space
        if point_prior is not None:
            if (
                len(point_prior.shape) != 1
                or point_prior.shape[0] != sim_space.shape[0]
            ):
                raise ValueError("Point priors not of correct size")
            self.point_prior = point_prior
        else:
            self.point_prior = np.array(
                [1.0 / sim_space.shape[0] for _ in range(sim_space.shape[0])]
            )

    def quazi_convexity(self, point_dist: np.ndarray, steps: int) -> float:
        if len(point_dist.shape) != 1:
            raise ValueError("Quazi-Convexity input must be a probability distribution")
        if np.size(point_dist) != self.sim_space.shape[0]:
            raise ValueError("Quazi-Convexity input must map to all points")
        if steps <= 0:
            raise ValueError("Steps must be positive")

        mesh = 1.0 / steps

        qc = 0

        steps = np.linspace(0, np.max(point_dist), steps)[::-1]

        for i in steps:
            # This is what the code from Skinner does, but this ignored colinear and coplanar points
            # TODO: handle those in the future
            level = self.sim_space[point_dist >= i]
            out_points = self.sim_space[point_dist < i]
            if self.sim_space.shape[1] == 1:
                flat = level.flatten()
                lo, hi = min(flat), max(flat)
                amount = level.shape[0]
                for p in out_points:
                    if p[0] <= hi and p[0] >= lo:
                        amount += 1
                qc += mesh * level.shape[0] / amount
            else:
                try:
                    hull = ConvexHull(level)
                    amount = level.shape[0]
                    # TODO: Vectorize probably
                    for point in out_points:
                        amount += int(
                            all(
                                (np.dot(eq[:-1], point) + eq[-1] <= 1e-12)
                                for eq in hull.equations
                            )
                        )
                    qc += mesh * level.shape[0] / amount
                except:
                    qc += mesh
                    pass
        return qc

    def encoder_convexity(
        self, encoder: np.ndarray, prior: np.ndarray, steps: int = 100
    ) -> float:
        # Apply Bayes' rule
        reconstructed = encoder.T * prior[:, None] / self.point_prior
        maximums = np.max(reconstructed, axis=0)

        reconstructed[~(reconstructed == maximums)] = 0
        reconstructed[reconstructed == maximums] = 1

        weighted_sum = np.sum(reconstructed.T, axis=0)
        weighted_sum = weighted_sum / np.sum(weighted_sum)

        convexities = []
        for word in encoder.T:
            convexities.append(self.quazi_convexity(word, steps))
        return np.sum(np.array(convexities) * weighted_sum)

    def language_convexity(self, lang: IBLanguage) -> float:
        return self.encoder_convexity(lang.qmw, lang.expressions_prior)
