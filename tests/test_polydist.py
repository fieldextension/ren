from fractions import Fraction

from ren.polydist import DiscreteGaussian, SparseTernary, Ternary


def test_key_dist_expected_squared_l2_norm():
    assert Ternary(n=8).expected_squared_l2_norm() == 4
    assert SparseTernary(n=8, h=3).expected_squared_l2_norm() == 3


def test_discrete_gaussian_one_norm_after_two_merges():
    dist = DiscreteGaussian(n=64, stdev=Fraction(1 << 18))
    # The base table ends at 626. At the two merges the weights are (13, 12) and (240, 239).
    assert dist.coeff_1_norm_bound() == 64 * 626 * (13 + 12) * (240 + 239)
