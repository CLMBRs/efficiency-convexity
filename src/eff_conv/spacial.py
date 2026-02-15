import numpy as np

class SimilarityFunction:
    @staticmethod
    def calculate_gradients(referent_locations: np.ndarray, word_locations: np.ndarray) -> np.ndarray:
        """
        Calculates gradients with respect to referent locations for given referent and word locations for
        the specified similarity function.

        Inputs:
        - referent_locations: ||U|| x L matrix, where U is the set of referents and L is the dimensionality of the space.
        - word_locations: ||W|| x L matrix, where W is the set of words and L is the dimensionality of the space.

        Output:
        - ||U|| x ||W|| x L matrix, the gradients for each of the referents and words.
        """
        raise NotImplementedError("SimilarityFunction is an abstract class, do not call calculate_gradients")


class CosineSimilarity(SimilarityFunction):
    @staticmethod
    def calculate_gradients(referent_locations: np.ndarray, word_locations: np.ndarray) -> np.ndarray:
        """
        Calculates the gradients for referents and words based on the derivative of the cosine similarity function
        for vectors.

        For inputs and outputs refer to SimilarityFunction.calculate_gradients

        The derivative this is based on is the following:
        d/dX cos(X, Y) = (Y - proj(X, Y))/(||X||*||Y||)
        """
        word_norms = np.linalg.norm(word_locations, axis=1)
        referent_norms = np.linalg.norm(referent_locations, axis=1)
        # Dot products between X and Y
        dot_products = (word_locations @ referent_locations.T).T
        # Projections from Y onto X
        # This is kept as a ||U|| x ||W|| x L matrix
        # Evil broadcasting gets this to work
        projections = referent_locations[:, None, :]*(dot_products/(referent_norms**2)[:, None])[:, :, None]
        subtractions = word_locations[None, :, :] - projections
        combined_norms = (word_norms*referent_norms[:, None])[:, :, None]
        
        return subtractions/combined_norms

def run_grad_step(
        referent_locations: np.ndarray,
        word_locations: np.ndarray,
        priors: np.ndarray,
        sim_func: SimilarityFunction,
) -> np.ndarray:
    """
    Recalculates referent locations based on gradient ascent of the following function: score(u) = sum_w q(u|w)*sim_func(u, w) 

    Inputs:
    - referent_locations: ||U|| x L matrix, where U is the set of referents and L is the dimensionality of the space.
    - word_locations: ||W|| x L matrix, where W is the set of words and L is the dimensionality of the space.
    - priors: The q(u|w) matrix, ||U|| x ||W||
    - sim_func: The similarity function used for scoring

    Output:
    - ||U|| x L matrix, the new positions for each referent.
    """

    # Check dimensions because it's very easy to get mixed up
    if len(referent_locations.shape) != 2 or len(word_locations.shape) != 2 or len(priors.shape) != 2:
        raise ValueError("Invalid shape for input array")
    ref_num, dimensionality = referent_locations.shape
    word_nums = word_locations.shape[0]
    if dimensionality != word_locations.shape[1]:
        raise ValueError(f"Conflicting dimensionality arguments: {dimensionality} and {word_locations.shape[1]}")
    if ref_num != priors.shape[0]:
        raise ValueError(f"Confliction referent numbers: {ref_num} and {priors.shape[0]}")
    if word_nums != priors.shape[1]:
        raise ValueError(f"Confliction word_nums numbers: {word_nums} and {priors.shape[1]}")
    
    # Gradients for every referent and word combination, weighted by priors
    # Is ||U|| x ||W|| x L
    weighted_gradients = sim_func.calculate_gradients(referent_locations, word_locations)*priors[:, :, None]

    return referent_locations + np.sum(weighted_gradients, axis=1)