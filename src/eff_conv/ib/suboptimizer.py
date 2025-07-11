from eff_conv.ib.language import IBLanguage
from eff_conv.ib.utils import drop_unused_dimensions

import math
import random


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
        range(shuffle_items.shape[0]),
        k=math.ceil(shuffle_items.shape[0] * shuffle_percent),
    )[:]
    unshuffled = selected[:]
    random.shuffle(selected)
    for u, s in zip(unshuffled, selected):
        shuffle_items[u] = lang.qwm.T[s]

    shuffle_items = shuffle_items.T

    # Drop unused dimensions
    shuffle_items = drop_unused_dimensions(shuffle_items)

    return IBLanguage(lang.structure, shuffle_items)
