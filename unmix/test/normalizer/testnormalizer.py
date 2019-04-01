#!/usr/bin/env python3
# coding: utf8

"""
Tests of the normalization and denormalization.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"


import numpy as np

from unmix.source.normalizers import normalizer


def test_normalize_denormalize():
    input = np.array([[[20, 1.5]]])
    normalized = normalizer.normalize(input)
    denormalized_complex = normalizer.denormalize(normalized)
    denormalized = np.zeros(shape=input.shape)
    denormalized[:, :, 0] = np.real(denormalized_complex)
    denormalized[:, :, 1] = np.imag(denormalized_complex)
    assert np.alltrue(denormalized == input)


if __name__ == "__main__":
    test_normalize_denormalize()
    print("Test run successful.")
