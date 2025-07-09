from dataclasses import dataclass
from functools import cached_property
from eff_conv.utils import IB_EPSILON, mutual_information

import numpy as np


@dataclass(frozen=True)
class IBStructure:
    """A structure is the probability mapping from meanings to referents and the priors for meanings.

    Properties:
        pum: A conditional probability matrix which maps from a meaning distribution to a referent distribution. Dimensions are ||R|| x ||M||.
        This matrix cannot have any 0s in it.

        meanings_prior: Probability distribution for the meanings. Length must be ||M||. Cannot have any 0s in it.
        If no value is passed in then a uniform distribution will be given.

        referents_prior: Probability distrubtions for the referents. Calculated via `pum` and `meanings_prior`. Length is ||R||

        mutual_information: Mutual information between referents and meanings. Formally I(U; M).
    """

    meanings_prior: np.ndarray
    pum: np.ndarray

    def __init__(
        self,
        pum: np.ndarray,
        prior: np.ndarray = None,
    ):
        if len(pum.shape) != 2:
            raise ValueError("Conditional probabilities must be a 2d matrix")
        if (np.abs(np.sum(pum, axis=0) - 1) > IB_EPSILON).any():
            raise ValueError(
                "All columns of conditional probability matrix must sum to 1"
            )
        if (pum <= 0).any():
            raise ValueError("Input matrix but all be greater than 0")

        if prior is not None:
            if pum.shape[1] != len(prior):
                raise ValueError(
                    f"Input matrix is for {pum.shape[1]} meanings, but {len(prior)} priors are given"
                )
            if (pum <= 0).any():
                raise ValueError("Priors must all be greater than 0")
            if abs(np.sum(prior)) - 1 > IB_EPSILON:
                raise ValueError("Priors must sum to 1")

        # If priors are not passed in, make it uniform
        object.__setattr__(
            self,
            "meanings_prior",
            (
                prior
                if prior is not None
                else np.array([1 / pum.shape[1] for _ in range(pum.shape[1])])
            ),
        )
        object.__setattr__(self, "pum", pum)

    @cached_property
    def referents_prior(self) -> np.ndarray:
        return self.pum @ self.meanings_prior

    @cached_property
    def mutual_information(self) -> float:
        return mutual_information(self.pum, self.referents_prior, self.meanings_prior)
