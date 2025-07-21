import numpy as np
import pytest

from eff_conv.convexity import SimilaritySpace
from eff_conv.ib.language import IBLanguage
from eff_conv.ib.structure import IBStructure
from eff_conv.ib.utils import IB_EPSILON


class TestConvexity:
    sim_space = SimilaritySpace(
        np.array(
            [
                # fmt: off
                [0, 0], [0, 1], [0, 2],
                [1, 0], [1, 1], [1, 2],
                [2, 0], [2, 1], [2, 2],
                # fmt: on
            ],
        )
    )
    sim_space_large = SimilaritySpace(
        np.array(
            [
                # fmt: off
                [0, 0], [0, 1], [0, 2], [0, 3],
                [1, 0], [1, 1], [1, 2], [1, 3],
                [2, 0], [2, 1], [2, 2], [2, 3],
                [3, 0], [3, 1], [3, 2], [3, 3],
                # fmt: on
            ],
        )
    )

    # Tests for convexity.py
    def test_similarity_space_check(self):
        with pytest.raises(ValueError):
            space_wrong_shape = SimilaritySpace(np.array([1]))
        with pytest.raises(ValueError):
            space_wrong_prior_len = SimilaritySpace(
                np.array([[1]]), point_prior=np.array([1, 2])
            )
        with pytest.raises(ValueError):
            space_wrong_prior_shape = SimilaritySpace(
                np.array([[1]]), point_prior=np.array([[1]])
            )
        with pytest.raises(ValueError):
            space_invalid_prior = SimilaritySpace(
                np.array([[1], [2]]), point_prior=np.array([1, 2])
            )
        with pytest.raises(ValueError):
            space_zero_prior = SimilaritySpace(
                np.array([[1], [2]]), point_prior=np.array([1, 0])
            )
        with pytest.raises(ValueError):
            space_negative_prior = SimilaritySpace(
                np.array([[1], [2]]), point_prior=np.array([2, -1])
            )

    def test_convexity_calculation(self):
        assert (
            abs(
                self.sim_space.encoder_convexity(np.ones((9, 1)) / 9, np.array([1])) - 1
            )
            < IB_EPSILON
        )
        # Epsilon is higher because rounding errors are more common beacuse of the nature of the algorithm
        assert (
            abs(
                self.sim_space.skinner_encoder_convexity(
                    np.array(
                        [
                            # fmt: off
                        [0.2, 0],
                        [0,   1],
                        [0.2, 0],
                        [0,   0],
                        [0.2, 0],
                        [0,   0],
                        [0.2, 0],
                        [0,   0],
                        [0.2, 0],
                            # fmt: on
                        ]
                    ),
                    np.array([0.5, 0.5]),
                )
                - 19 / 27
            )
            < 0.005
        )
        assert (
            abs(
                self.sim_space.encoder_convexity(
                    np.array(
                        [
                            # fmt: off
                        [0.2, 0],
                        [0,   1],
                        [0.2, 0],
                        [0,   0],
                        [0.2, 0],
                        [0,   0],
                        [0.2, 0],
                        [0,   0],
                        [0.2, 0],
                            # fmt: on
                        ]
                    ),
                    np.array([0.5, 0.5]),
                )
                - 14 / 18
            )
            < 0.005
        )

    def test_projection_down(self):
        # This also tests 1d spaces
        assert (
            abs(
                self.sim_space_large.encoder_convexity(
                    np.array(
                        [
                            # fmt: off
                        [0.33], [0], [0], [0],
                        [0],    [0], [0], [0],
                        [0.33], [0], [0], [0],
                        [0.33], [0], [0], [0],
                            # fmt: on
                        ]
                    )
                    / 0.99,
                    np.array([1]),
                )
                - 3 / 4
            )
            < 0.005
        )
        assert (
            abs(
                self.sim_space_large.encoder_convexity(
                    np.array(
                        [
                            # fmt: off
                        [0.5], [0], [0], [0],
                        [0],   [0], [0], [0],
                        [0],   [0], [0], [0],
                        [0.5], [0], [0], [0],
                            # fmt: on
                        ]
                    ),
                    np.array([1]),
                    steps=1000,
                )
                - 1 / 2
            )
            < 0.005
        )

    def test_language_convexity(self):
        simple_struct = IBStructure(
            np.array(
                [
                    [0.465, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.465]
                    for _ in range(9)
                ]
            ).T
        )
        simple_lang = IBLanguage(simple_struct, np.ones((1, 9)))
        assert (
            abs(
                self.sim_space.language_convexity(simple_lang)
                - self.sim_space.encoder_convexity(np.ones((9, 1)) / 9, np.array([1]))
            )
            < IB_EPSILON
        )
        # This will fail on remote if the epsilon is not higher
        assert (
            abs(
                self.sim_space.language_convexity(
                    simple_lang, referents=True, steps=1000
                )
                - self.sim_space.encoder_convexity(
                    np.array(
                        [[0.465, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.465]]
                    ).T,
                    np.array([1]),
                    steps=1000,
                )
            )
            < 0.005
        )

    def test_convexity_check(self):
        with pytest.raises(ValueError):
            input_wrong_shape = self.sim_space.quasi_convexity(np.ones((9, 1)), 100)
        with pytest.raises(ValueError):
            input_wrong_size = self.sim_space.quasi_convexity(np.ones(8) / 8, 100)
        with pytest.raises(ValueError):
            input_invalid = self.sim_space.quasi_convexity(np.ones(9), 100)
        with pytest.raises(ValueError):
            input_negative = self.sim_space.quasi_convexity(
                np.array([-7, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 100
            )
        with pytest.raises(ValueError):
            steps_negative = self.sim_space.quasi_convexity(np.ones(9) / 9, -1)

    def test_encoder_check(self):
        with pytest.raises(ValueError):
            prior_invalid = self.sim_space.encoder_convexity(
                np.ones((9, 1)) / 9, np.array([0.9])
            )
        with pytest.raises(ValueError):
            prior_invalid_skinner = self.sim_space.skinner_encoder_convexity(
                np.ones((9, 1)) / 9, np.array([0.9])
            )
        with pytest.raises(ValueError):
            prior_negative = self.sim_space.encoder_convexity(
                np.ones((9, 2)) / 9, np.array([-1, 2])
            )
        with pytest.raises(ValueError):
            prior_negative_skinner = self.sim_space.skinner_encoder_convexity(
                np.ones((9, 2)) / 9, np.array([-1, 2])
            )
