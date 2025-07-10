import numpy as np

from eff_conv.ib.language import IBLanguage
from eff_conv.ib.structure import IBStructure
from eff_conv.ib.optimization import normals, recalculate_language


class TestOptimization:
    simple_struct = IBStructure(np.array([[0.5, 0.5], [0.5, 0.5]]))
    simple_lang = IBLanguage(simple_struct, np.array([[1, 1]]))
    complex_lang = IBLanguage(simple_struct, np.identity(2))

    # Tests for optimization.py
    def test_normals_calculation(self):
        assert np.array_equal(normals(self.simple_lang, 1), np.array([1, 1]))
        assert np.array_equal(normals(self.complex_lang, 1), np.array([1, 1]))

    def test_recalculate_language(self):
        recalculated = recalculate_language(self.simple_lang, 1)
        assert np.array_equal(self.simple_lang.qwm, recalculated.qwm)
        recalculated = recalculate_language(self.complex_lang, 1)
        assert np.array_equal(recalculated.qwm, np.array([[0.5, 0.5], [0.5, 0.5]]))
