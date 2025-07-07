from eff_conv.language import IBLanguage
from eff_conv.utils import IB_EPSILON

from math import ceil

import random
import numpy as np


def shuffle_language(lang: IBLanguage, shuffle_percent: float) -> IBLanguage:
    shuffle_items = lang.qwm.T[:]
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
