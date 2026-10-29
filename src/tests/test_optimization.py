import numpy as np
import pytest

from eff_conv.ib.language import IBLanguage
from eff_conv.ib.structure import IBStructure
from eff_conv.ib.optimization import (
    calculate_optimal,
    normals,
    recalculate_language,
)


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

    # A structure with graded meanings, where BA converges slowly near phase transitions
    graded_pum = np.exp(-0.5 * (np.arange(6)[:, None] - np.arange(6)[None, :]) ** 2)
    graded_struct = IBStructure(graded_pum / graded_pum.sum(axis=0))

    @staticmethod
    def objective(language, beta):
        return language.complexity - beta * language.iwu

    @pytest.mark.parametrize("beta", [1.5, 2.5, 4.0, 8.0])
    @pytest.mark.parametrize(
        "tol_objective, tol_encoder", [(1e-6, 1e-5), (1e-7, None), (None, 1e-6)]
    )
    def test_calculate_optimal_is_stationary(self, beta, tol_objective, tol_encoder):
        seed = IBLanguage(self.graded_struct, np.identity(6))
        language = calculate_optimal(
            self.graded_struct, beta, seed, tol_objective, tol_encoder
        )
        # Running many more BA iterations should barely improve the objective
        continued = language
        for _ in range(2000):
            continued = recalculate_language(continued, beta)
        assert self.objective(language, beta) - self.objective(continued, beta) < 1e-6

    def test_calculate_optimal_fixed_point(self):
        # The fully uninformative language is a fixed point: stop right away, unchanged
        language = calculate_optimal(self.simple_struct, 1, self.simple_lang, max_iter=1)
        assert np.array_equal(language.qwm, self.simple_lang.qwm)

    def test_calculate_optimal_warns_without_convergence(self):
        seed = IBLanguage(self.graded_struct, np.identity(6))
        with pytest.warns(RuntimeWarning, match="did not converge"):
            calculate_optimal(self.graded_struct, 2.5, seed, max_iter=2)

    def test_calculate_optimal_needs_a_criterion(self):
        with pytest.raises(ValueError):
            calculate_optimal(
                self.simple_struct, 1, self.simple_lang, tol_objective=None, tol_encoder=None
            )
