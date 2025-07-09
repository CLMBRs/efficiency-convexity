from math import log2
import pytest
import numpy as np

from eff_conv.language import IBLanguage
from eff_conv.structure import IBStructure
from eff_conv.utils import (
    IB_EPSILON,
    generate_random_expressions,
    kl_divergence,
    mutual_information,
    safe_log,
)


class TestIB:

    simple_struct = IBStructure(np.array([[0.5, 0.5], [0.5, 0.5]]))
    simple_lang = IBLanguage(simple_struct, np.array([[1, 1]]))
    complex_lang = IBLanguage(simple_struct, np.identity(2))

    # Tests for structure.py
    def test_meaning_priors(self):
        assert np.array_equal(self.simple_struct.meanings_prior, np.array([0.5, 0.5]))

    def test_referent_priors(self):
        assert np.array_equal(self.simple_struct.referents_prior, np.array([0.5, 0.5]))

    def test_structure_mutual_information(self):
        assert self.simple_struct.mutual_information == 0

    def test_ib_structure_check(self):
        with pytest.raises(ValueError):
            struct_wrong_shape = IBStructure(np.array([]))
        with pytest.raises(ValueError):
            struct_not_probability = IBStructure(np.array([[1, 2, 3]]))
        with pytest.raises(ValueError):
            struct_with_zero = IBStructure(np.array([[0, 1, 0]]))
        with pytest.raises(ValueError):
            struct_with_negative = IBStructure(np.array([[1, -1, 1]]))
        with pytest.raises(ValueError):
            struct_wrong_prior = IBStructure(
                np.array([[0.25, 0.25, 0.25, 0.25]]), np.array([])
            )
        with pytest.raises(ValueError):
            struct_zero_prior = IBStructure(
                np.array([[0.25, 0.25, 0.25, 0.25]]), np.array([0, 0, 1, 0])
            )
        with pytest.raises(ValueError):
            struct_negative_prior = IBStructure(
                np.array([[0.25, 0.25, 0.25, 0.25]]), np.array([.25, .25, -0.5, 1])
            )
        with pytest.raises(ValueError):
            struct_invalid_prior = IBStructure(
                np.array([[0.25, 0.25, 0.25, 0.25]]), np.array([1, 2, 3, 4])
            )

    # Tests for utils.py
    def test_safe_log(self):
        assert np.array_equal(safe_log(np.ones((2, 2)) * 2), np.ones((2, 2)))
        assert np.array_equal(safe_log(np.zeros((2, 2))), np.zeros((2, 2)))

    def test_random_expressions(self):
        assert generate_random_expressions(10).shape == (10, 10)
        assert (
            np.abs(np.sum(generate_random_expressions(10), axis=0) - 1) < IB_EPSILON
        ).all()

    def test_kl_divergence(self):
        with pytest.raises(ValueError):
            invalid_arr1 = kl_divergence(np.array([0]), np.array([1]))
        with pytest.raises(ValueError):
            invalid_arr2 = kl_divergence(np.array([1]), np.array([0]))
        assert kl_divergence(np.array([1]), np.array([1])) == 0
        expected = 0.75 * log2(3 / 2) - 0.25
        assert (
            abs(kl_divergence(np.array([0.25, 0.75]), np.array([0.5, 0.5])) - expected)
            < IB_EPSILON
        )
        expected = 0.5 * log2(2 / 3) + 0.5
        assert (
            abs(kl_divergence(np.array([0.5, 0.5]), np.array([0.25, 0.75])) - expected)
            < IB_EPSILON
        )

    def test_mutual_information(self):
        with pytest.raises(ValueError):
            invalid_px = mutual_information(
                np.array([[1]]), np.array([0]), np.array([1])
            )
        with pytest.raises(ValueError):
            invalid_py = mutual_information(
                np.array([[1]]), np.array([0]), np.array([1])
            )
        with pytest.raises(ValueError):
            invalid_pxy = mutual_information(
                np.array([[0]]), np.array([1]), np.array([1])
            )
        with pytest.raises(ValueError):
            invalid_px_shape = mutual_information(
                np.array([[0.5, 0.5]]), np.array([0.5, 0.5]), np.array([0.5, 0.5])
            )
        with pytest.raises(ValueError):
            invalid_py_shape = mutual_information(
                np.array([[0.5, 0.5]]), np.array([1]), np.array([1])
            )
        assert (
            mutual_information(np.array([[1, 1]]), np.array([1]), np.array([0.5, 0.5]))
            == 0
        )
        expected = 0.25 * log2(1 / 2) + 0.75 * log2(3 / 2)
        assert (
            mutual_information(
                np.array([[0.25, 0.25], [0.75, 0.75]]),
                np.array([0.5, 0.5]),
                np.array([0.5, 0.5]),
            )
            == expected
        )

    # Tests for language.py
    def test_expressions_prior(self):
        assert np.array_equal(self.complex_lang.expressions_prior, np.array([0.5, 0.5]))
        assert np.array_equal(self.simple_lang.expressions_prior, np.array([1]))

    def test_complexity(self):
        assert self.complex_lang.complexity == 1
        assert self.simple_lang.complexity == 0

    def test_complexity(self):
        assert self.complex_lang.complexity == 1
        assert self.simple_lang.complexity == 0

    def test_reconstructed_meanings(self):
        assert np.array_equal(
            self.complex_lang.reconstructed_meanings, self.simple_struct.pum
        )
        assert np.array_equal(
            self.simple_lang.reconstructed_meanings, np.array([[0.5], [0.5]])
        )

    def test_divergence_array(self):
        assert np.array_equal(self.complex_lang.divergence_array, np.zeros((2, 2)))
        assert np.array_equal(self.simple_lang.divergence_array, np.zeros((1, 2)))

    def test_expected_divergence(self):
        assert self.complex_lang.expected_divergence == 0
        assert self.simple_lang.expected_divergence == 0

    def test_expected_divergence(self):
        assert self.complex_lang.iwu == 0
        assert self.simple_lang.iwu == 0
        assert (
            self.simple_struct.mutual_information - self.complex_lang.iwu
            == self.complex_lang.expected_divergence
        )
        assert (
            self.simple_struct.mutual_information - self.simple_lang.iwu
            == self.simple_lang.expected_divergence
        )

    def test_ib_language_check(self):
        with pytest.raises(ValueError):
            lang_wrong_shape = IBLanguage(self.simple_struct, np.array([1]))
        with pytest.raises(ValueError):
            lang_wrong_shape2 = IBLanguage(self.simple_struct, np.array([[1]]))
        with pytest.raises(ValueError):
            lang_invalid_probability = IBLanguage(
                self.simple_struct, np.array([[2, 2]])
            )
        with pytest.raises(ValueError):
            lang_negative_probability = IBLanguage(
                self.simple_struct, np.array([[-1, 1], [2, 0]])
            )
