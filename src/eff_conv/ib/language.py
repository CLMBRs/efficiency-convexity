from functools import cached_property
from eff_conv.ib.structure import IBStructure
from eff_conv.ib.utils import IB_EPSILON, kl_divergence, mutual_information

import numpy as np


class IBLanguage:
    """A language has expressions which are mapped to from meanings and which can map to expressions.

    Properties:
        structure: This is the structure in which the language exists.

        qwm: This is a conditional probaiblity matrix which maps a meaning distribution to expressions. Dimensions are ||W|| x ||M||.
        Note: The columns of the matrix are the probability distributions. This differs from other implementations.

        qmw: Reconstructed conditional probability matrix which maps an expression distrubution to meanings. Created using Bayes' rule.
        Dimensions are ||M|| x ||W||.

        complexity: Mutual information between expressions and meanings. Formally I(W; M).

        expressions_prior: Probability distribution for expressions. Constructed from the structure's meaning priors and qwm. Formally p(w).

        reconstructed_meanings: Conditional probability matrix which maps an expression distrubition to referents. Created using qmw and structure.pum.
        Dimensions are ||U|| x ||W||.

        divergence_array: Matrix which stores the different KL Divergences between the referent probability distrubutions per meaning and per expression.
        Dimensions are ||W|| x ||M||. (It is important to note that the KL Divergence function uses base 2 logarithms)

        expected_divergence: This is the expected KL Divergence between the language's reconstructed meanings and the structure's meanings.
        expected divergence = I(U; M) - I(W; U)

        iwu: The mutual information between the expressions of a language and the referents. Also referred to as accuracy. Formally I(W; U)
    """

    structure: IBStructure
    qwm: np.ndarray

    def __init__(
        self,
        structure: IBStructure,
        qwm: np.ndarray,
    ):
        if len(qwm.shape) != 2:
            raise ValueError("Must be a 2d matrix")
        if qwm.shape[1] != structure.pum.shape[1]:
            raise ValueError(
                f"Input matrix is for {qwm.shape[1]} meanings, not {structure.pum.shape[1]}"
            )
        if (np.abs(np.sum(qwm, axis=0) - 1) > IB_EPSILON).any():
            raise ValueError(
                "All columns of conditional probability matrix must sum to 1"
            )
        if (qwm < 0).any():
            raise ValueError(
                "No negative numbers are allowed in the probability matrix"
            )
        self.structure = structure
        self.qwm = qwm

    @cached_property
    def qmw(self) -> np.ndarray:
        # Apply Bayes' rule
        return (
            self.qwm.T * self.structure.meanings_prior[:, None] / self.expressions_prior
        )

    @cached_property
    def complexity(self) -> float:
        return mutual_information(
            self.qwm, self.expressions_prior, self.structure.meanings_prior
        )

    @cached_property
    def expressions_prior(self) -> np.ndarray:
        # Normalization does become important at really small values
        intermediate = self.qwm @ self.structure.meanings_prior
        return intermediate / np.sum(intermediate)

    @cached_property
    def reconstructed_meanings(self) -> np.ndarray:
        # Normalization does become important at really small values
        intermediate = self.structure.pum @ self.qmw
        return intermediate / np.sum(intermediate, axis=0)

    @cached_property
    def divergence_array(self) -> np.ndarray:
        return np.array(
            [
                [kl_divergence(k, r) for k in self.structure.pum.T]
                for r in self.reconstructed_meanings.T
            ]
        )

    @cached_property
    def expected_divergence(self) -> float:
        left = self.qwm * self.structure.meanings_prior
        return np.sum(left * self.divergence_array)

    @cached_property
    def iwu(self) -> float:
        return mutual_information(
            self.reconstructed_meanings,
            self.structure.referents_prior,
            self.expressions_prior,
        )
