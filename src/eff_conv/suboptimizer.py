from eff_conv.language import IBLanguage
from eff_conv.utils import IB_EPSILON

from math import ceil

import random
import numpy as np


def shuffle_language(lang: IBLanguage, shuffle_percent: float) -> IBLanguage:
    """Shuffles the columns of the language's Q(w|m) matrix in order to produce suboptimal music. The percentage of columns shuffled can be adjusted
    This comes from Skinner L. (2025).

    Args:
        lang (IBLanguage): The language to shuffle
        shuffle_percent (float): The percentage of columns to shuffle (Range: [0, 1])
    Returns:
        tuple[tuple[IBLanguage, float], ...]: Languages and their respective beta values.
    """
    if shuffle_percent < 0 or shuffle_percent > 1:
        raise ValueError("`shuffle_percent` must be between 0 and 1")
    shuffle_items = lang.qwm.T.copy()
    selected = random.choices(
        range(shuffle_items.shape[0]), k=ceil(shuffle_items.shape[0] * shuffle_percent)
    )[:]
    unshuffled = selected[:]
    random.shuffle(selected)
    for u, s in zip(unshuffled, selected):
        shuffle_items[u] = lang.qwm.T[s]

    shuffle_items = shuffle_items.T
    # Drop unused dimensions
    shuffle_items = shuffle_items[~np.all(shuffle_items <= IB_EPSILON, axis=1)]
    # Normalize just because of floating point issues
    shuffle_items /= np.sum(shuffle_items, axis=0)
    return IBLanguage(lang.structure, shuffle_items)
