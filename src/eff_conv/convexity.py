from eff_conv.ib.language import IBLanguage

from scipy.spatial import ConvexHull

import numpy as np

from eff_conv.ib.utils import IB_EPSILON


class SimilaritySpace:
    """A similarity space contains points (which should correspond in order to referents or meanings) and the priors upon those points.

    Properties:
        sim_space: A matrix which stores a list of points, each point should correspond to a referent or meaning. Dimensions are D x ||P||.
        Where D is the dimension of the points and P is the set of points.

        point_prior: Probability distribution for the points. Length must be ||P||. Cannot have any 0s in it.
        If no value is passed in then a uniform distribution will be given.
    """

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
            if np.abs(np.sum(point_prior) - 1) > IB_EPSILON:
                raise ValueError("Point priors must sum to 1")
            if (point_prior <= 0).any():
                raise ValueError("Prior values must be greater than 0")
            self.point_prior = point_prior
        else:
            self.point_prior = np.array(
                [1.0 / sim_space.shape[0] for _ in range(sim_space.shape[0])]
            )

    def _1d_convexity_amount(self, points: np.ndarray, level: np.ndarray) -> int:
        """Finds the number of points which are contained within a 1d range.

        Args:
            points (np.ndarray): The points to check.
            level (np.ndarray): The points which make up the 1d range

        Returns:
            int: The number of points which are in the range spanned by levels.
        """
        l_flat = level.flatten()
        p_flat = points.flatten()
        lo, hi = min(l_flat), max(l_flat)
        return sum((p_flat <= hi) & (p_flat >= lo)) + level.shape[0]

    def quasi_convexity(self, point_dist: np.ndarray, steps: int) -> float:
        """Finds the quasi-convexity of a probability. Algorithm from Skinner L. (2025).

        Args:
            point_dist (np.ndarray): The probability distribution to be evaluated.
            steps (int): The number of steps to interate over the probability (higher is more accurate but slower)

        Returns:
            float: The quasi-convexity of the probabilty distribution.
        """

        if len(point_dist.shape) != 1:
            raise ValueError("Quasi-Convexity input must be a probability distribution")
        if np.size(point_dist) != self.sim_space.shape[0]:
            raise ValueError("Quasi-Convexity input must map to all points")
        if np.abs(np.sum(point_dist) - 1) > IB_EPSILON:
            raise ValueError("Quasi-Convexity input must sum to 1")
        if (point_dist < 0).any():
            raise ValueError("Quasi-Convexity input must be greater than or equal to 0")
        if steps <= 0:
            raise ValueError("Steps must be positive")

        mesh = 1.0 / steps

        qc = 0

        steps = np.linspace(0, np.max(point_dist), steps)[::-1]

        for i in steps:
            level = self.sim_space[point_dist >= i]
            out_points = self.sim_space[point_dist < i]
            # If everything is on a line a very simple calculation can be done
            if self.sim_space.shape[1] == 1:
                qc += (
                    mesh * level.shape[0] / self._1d_convexity_amount(out_points, level)
                )
            else:
                # See if the points don't span the space (If so, ConvexHull will throw an error)
                consider = out_points
                rank = np.linalg.matrix_rank(level - level[0])
                if rank < self.sim_space.shape[1]:
                    consider = []
                    for p in out_points:
                        check = np.concatenate((level, [p]))
                        if rank == np.linalg.matrix_rank(check - check[0]):
                            consider.append(p)

                    # Project down
                    U, _, _ = np.linalg.svd(level.T, full_matrices=False)
                    proj = U[:, :rank].T
                    if len(consider) > 0:
                        consider = (proj @ np.array(consider).T).T
                    else:
                        qc += mesh
                        continue
                    level = (proj @ level.T).T

                if rank == 1:
                    amount = self._1d_convexity_amount(consider, level)
                else:
                    hull = ConvexHull(level)
                    eqs = hull.equations[:, :-1]
                    end = hull.equations[:, -1]
                    amount = level.shape[0] + sum(
                        np.all(eqs @ consider.T + end[:, None] <= 1e-12, axis=0)
                    )

                qc += mesh * level.shape[0] / amount
        return qc

    def skinner_encoder_convexity(
        self, distrubitions: np.ndarray, prior: np.ndarray, steps: int = 100
    ) -> float:
        """Finds the quasi-convexity of a conditional probabilty matrix, typically an IB encoder. Weighting algorithm from Skinner L. (2025).

        Args:
            distrubitions (np.ndarray): The conditional probaility matrix to be evaluated. Shape is of ||P|| x n where n > 0.
            Each column of the matrix should be a probability distrubtion over P.

            prior (np.ndarray): The probability distribution of inputs into the encoder. Must be of size n.
            steps (int, default: 100): The number of steps to interate over the probability (higher is more accurate but slower)

        Returns:
            float: The quasi-convexity of the matrix.
        """

        if np.abs(np.sum(prior) - 1) > IB_EPSILON:
            raise ValueError("Prior must sum to 1")
        if (prior <= 0).any():
            raise ValueError("Prior must be greater than 0")

        # Apply Bayes' rule
        reconstructed = distrubitions.T * prior[:, None] / self.point_prior
        maximums = np.max(reconstructed, axis=0)

        reconstructed[~(reconstructed == maximums)] = 0
        reconstructed[reconstructed == maximums] = 1

        weighted_sum = np.sum(reconstructed.T, axis=0)
        weighted_sum = weighted_sum / np.sum(weighted_sum)

        convexities = []
        for word in distrubitions.T:
            convexities.append(self.quasi_convexity(word, steps))
        return np.sum(np.array(convexities) * weighted_sum)

    def encoder_convexity(
        self, distrubitions: np.ndarray, prior: np.ndarray, steps: int = 100
    ) -> float:
        """Finds the quasi-convexity of a conditional probabilty matrix, typically an IB encoder. Weighting is based on the p(y) for p(x|y).

        Args:
            distrubitions (np.ndarray): The conditional probaility matrix to be evaluated. Shape is of ||P|| x n where n > 0.
            Each column of the matrix should be a probability distrubtion over P.

            prior (np.ndarray): The probability distribution of inputs into the encoder. Must be of size n.
            steps (int, default: 100): The number of steps to interate over the probability (higher is more accurate but slower)

        Returns:
            float: The quasi-convexity of the matrix.
        """

        if np.abs(np.sum(prior) - 1) > IB_EPSILON:
            raise ValueError("Prior must sum to 1")
        if (prior <= 0).any():
            raise ValueError("Prior must be greater than 0")

        convexities = []
        for word in distrubitions.T:
            convexities.append(self.quasi_convexity(word, steps))
        return np.sum(np.array(convexities) * prior)

    def language_convexity(
        self, lang: IBLanguage, steps: int = 100, referents=False
    ) -> float:
        """Finds the quasi-convexity of an IBLanguage by evaluating the Q(m|w) matrix.

        Args:
            lang (IBLanguage): The language to be evaluated.
            steps (int, default: 100): The number of steps to iterate over the probability (higher is more accurate but slower).
            refeernts (boolean, default: False): Whether to evaluate q(u|w) or q(m|w).

        Returns:
            float: The quasi-convexity of the language.
        """

        return self.encoder_convexity(
            lang.reconstructed_meanings if referents else lang.qmw,
            lang.expressions_prior,
            steps=steps,
        )
