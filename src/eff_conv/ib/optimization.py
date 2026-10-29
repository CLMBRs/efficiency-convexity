from eff_conv.ib.language import IBLanguage
from eff_conv.ib.structure import IBStructure
from eff_conv.ib.utils import (
    drop_unused_dimensions,
    generate_random_expressions,
)

import numpy as np
import multiprocessing as mp
import math
import warnings

LOG_2 = math.log(2)


# Calculate the normal function results for the meanings
def normals(language: IBLanguage, beta: float) -> np.ndarray:
    """Calculates the normals (Z(x, b)) for a given language.
    NOTE: This will break if beta is too high and the language is not near fully complex (i.e. language.qwm is not nearing the identity matrix).
    This is because language.divergence_array will have extremely high values which will cause the exponentiation to go down to a full 0 array.

    Args:
        language (IBLanguage): Language to calculate the normals of
        beta (float): Beta to calculate the normal at

    Returns:
        np.ndarray: Array of normals for each meaning in the structure .
    """
    return np.sum(
        np.exp(-beta * language.divergence_array * LOG_2)
        * language.expressions_prior[:, None],
        axis=0,
    )


def recalculate_language(language: IBLanguage, beta: float) -> IBLanguage:
    """Run an iteration of the BA algorithm to get the language closer to the optimal language for the given beta
    NOTE: As with normals this function will break if beta is too high and the language is not nearly fully complex (i.e. language.qwm is not nearing the identity matrix).
    This is because normals will return a matrix filled with 0s, which will invalidate the math inside the function.

    Args:
        language (IBLanguage): Language to run the BA algorithm on
        beta (float): Beta to calculate the normal at

    Returns:
        IBLanguage: An updated language which is closer to the optimal language.
    """
    # Recalculate q(w|m)
    recalculated_qwm = (
        language.expressions_prior[:, None]
        * np.exp(-beta * language.divergence_array * LOG_2)
        / normals(language, beta)
    )

    # Drop unused dimensions
    recalculated_qwm = drop_unused_dimensions(recalculated_qwm)

    # Create new language
    return IBLanguage(
        language.structure,
        recalculated_qwm,
    )


def calculate_optimal(
    structure: IBStructure,
    beta: float,
    start: IBLanguage = None,
    tol_objective: float = 1e-6,
    tol_encoder: float = 1e-5,
    max_iter: int = 10000,
) -> IBLanguage:
    """Find the optimal language for a given beta given a structure
    NOTE: As with recalculate_language this function will break if beta is too high and the seed language is not nearly fully complex (i.e. language.qwm is not nearing the identity matrix).
    This is because recalculate_language will break under these conditions (see notes for it).
    NOTE: This is not guaranteed to find the exact optimal language, as it can get stuck near the optimal frontier. It may be based to use run_deterministic_annealing

    Iteration stops when both of the following hold:
    - the estimated remaining decrease of the objective is at most tol_objective. The BA step
      never increases the objective, and it converges roughly geometrically, so if the last two
      decreases were d_prev and d, the remaining decrease is about d * r / (1 - r) with
      r = d / d_prev. Near phase transitions r is close to 1 and this is much larger than d.
    - no entry of q(w|m) changed by more than tol_encoder in the last step.
    If that does not happen within max_iter iterations a RuntimeWarning is raised and the last
    language is returned.

    Args:
        structure (IBStructure): The structure for which the language will be optimized
        beta (float): Beta to calculate the optimal at
        start (IBLanguage, optional): A starting point for the optimizer. If not passed in a random language will be generated and used
        tol_objective (float, default 1e-6): Tolerance on the estimated remaining decrease of the objective
        tol_encoder (float, default 1e-5): Tolerance on the change of q(w|m) in one iteration
        max_iter (int, default 10000): Maximum number of BA iterations

    Returns:
        IBLanguage: An updated language which is closer to the optimal language.
    """
    language = (
        start
        if start is not None
        else IBLanguage(structure, generate_random_expressions(structure.pum.shape[1]))
    )

    objective = language.complexity - beta * language.iwu
    prev_decrease = None

    for _ in range(max_iter):
        new_language = recalculate_language(language, beta)
        new_objective = new_language.complexity - beta * new_language.iwu
        decrease = abs(objective - new_objective)

        # Estimated remaining decrease of the objective (Aitken extrapolation)
        if decrease == 0:
            remaining = 0.0
        elif prev_decrease is None or decrease >= prev_decrease:
            remaining = math.inf
        else:
            rate = decrease / prev_decrease
            remaining = decrease * rate / (1 - rate)

        # Words may be dropped by recalculate_language, in which case the encoder has changed
        encoder_settled = (
            new_language.qwm.shape == language.qwm.shape
            and np.abs(new_language.qwm - language.qwm).max() <= tol_encoder
        )

        language, objective, prev_decrease = new_language, new_objective, decrease
        if remaining <= tol_objective and encoder_settled:
            return language

    warnings.warn(
        f"BA did not converge at beta={beta} within {max_iter} iterations",
        RuntimeWarning,
    )
    return language


# Modified from embo/Lindsay Skinner's code
def get_optimial_languages(
    structure: IBStructure, start: float, end: float, steps: int, threads: int = 1
) -> tuple[tuple[IBLanguage, float], ...]:
    """Finds the optimal languages for a given range of beta values, can be multithreaded

    Args:
        structure (IBStructure): The structure for which the language will be optimized
        start (float): Start of the beta range
        end (float): End of the beta range
        steps (int): The number of betas to calculate
        threads (int, default 1): The number of threads to parallel process

    Returns:
        tuple[tuple[IBLanguage, float], ...]: Languages and their respective beta values.
    """
    # Get beta values
    beta_vec = np.linspace(start, end, steps)

    # Parallel computing of compression for desired beta values
    with mp.Pool(processes=threads) as pool:
        results = [
            pool.apply_async(calculate_optimal, args=(structure, b)) for b in beta_vec
        ]
        langs = tuple(p.get() for p in results)
    return tuple(zip(langs, beta_vec))


def run_deterministic_annealing(
    structure: IBStructure,
    betas: tuple[float, ...],
    verbose: bool = False,
    seed: IBLanguage = None,
    tol_objective: float = 1e-6,
    tol_encoder: float = 1e-5,
    max_iter: int = 10000,
) -> tuple[tuple[IBLanguage, float], ...]:
    """Run forward or reverse deterministic annealing approach. This means that it will start either with a very high value of beta and work down (reverse)
    or a very low value of beta and work up (forward). When going to a new value of beta it will seed the optimizer with the output of the previous beta iteration
    in an attempt to avoid getting stuck on local maxima.

    For reverse deterministic annealing betas can start at around 2^13, and for forward they can start at 0. Betas should not be evenly spaced.

    NOTE: If doing reverse deterministic annealing you should seed the function with a near fully complex language (i.e. one where the qwm is near the idenity matrix).
    For more information on why see the notes for calculate_optimial

    Args:
        structure (IBStructure): The structure for which the languages will be optimized
        betas (float): The betas to optimize for, this will be iterated through in order
        verbose (bool): Output to console when languages converge
        start (IBLanguage, optional): A starting point for the annealing run. If not passed in a random language will be generated and used
        tol_objective, tol_encoder, max_iter: Stopping criteria passed to calculate_optimal at every beta

    Returns:
        tuple[tuple[IBLanguage, float], ...]: Languages and their respective beta values.
    """
    tolerances = dict(
        tol_objective=tol_objective, tol_encoder=tol_encoder, max_iter=max_iter
    )
    languages = [calculate_optimal(structure, betas[0], seed, **tolerances)]

    if verbose:
        print(f"Beta {betas[0]} converged (1/{len(betas)})")

    for i in range(1, len(betas)):
        languages.append(
            calculate_optimal(structure, betas[i], languages[i - 1], **tolerances)
        )
        if verbose:
            print(f"Beta {betas[i]} converged ({i+1}/{len(betas)})")

    return tuple(zip(languages, betas))
