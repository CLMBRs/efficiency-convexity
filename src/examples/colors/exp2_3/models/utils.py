from eff_conv.convexity import SimilaritySpace

import numpy as np


class ModelSystem:
    def __init__(self):
        raise NotImplementedError("ModelSystem.__init__ should not be called")

    def generate_meanings(self) -> np.ndarray:
        """
        Generates the meanings array utilizing the distribution function which was passed in

        Returns:
            np.ndarray: The meanings matrix which can be passed into a IBStructure
        """
        raise NotImplementedError("ModelSystem.generate_meanings should not be called")

    def meaning_priors(self) -> np.ndarray:
        raise NotImplementedError("ModelSystem.meaning_priors should not be called")

    def space(self) -> SimilaritySpace:
        """
        Generates a similarity space for the system

        Returns:
            SimilaritySpace: The similarity space for the system
        """
        raise NotImplementedError("ModelSystem.range should not be called")

    def name(self) -> str:
        """
        Provides an identification string for the system

        Returns:
            str: The identification string for the system
        """
        raise NotImplementedError("ModelSystem.name should not be called")
