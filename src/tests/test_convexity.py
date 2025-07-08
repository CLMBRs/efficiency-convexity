import numpy as np

from eff_conv.convexity import SimilaritySpace
from eff_conv.language import IBLanguage
from eff_conv.structure import IBStructure
from eff_conv.utils import IB_EPSILON


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
    def test_convexity_calculation(self):
        assert (
            self.sim_space.encoder_convexity(np.ones((9, 1)) / 9, np.array([1])) - 1
            < IB_EPSILON
        )
        # Epsilon is higher because rounding errors are more common beacuse of the nature of the algorithm
        assert (
            self.sim_space.encoder_convexity(
                np.array([[0.2], [0], [0.2], [0], [0.2], [0], [0.2], [0], [0.2]]),
                np.array([1]),
            )
            - 5 / 9
            < 0.005
        )

    def test_projection_down(self):
        # This also tests 1d spaces
        assert (
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
                ),
                np.array([1]),
            )
            - 3 / 4
            < 0.005
        )
        assert (
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
            < 0.005
        )

    def test_language_convexity(self):
        simple_struct = IBStructure(np.ones((9, 9)) / 9)
        simple_lang = IBLanguage(simple_struct, np.ones((1, 9)))
        assert (
            self.sim_space.language_convexity(simple_lang)
            - self.sim_space.encoder_convexity(np.ones((9, 1)) / 9, np.array([1]))
            < IB_EPSILON
        )
