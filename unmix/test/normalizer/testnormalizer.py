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
    input = np.array([[[[0.1], [1j]]]])
    normalized = normalizer.normalize(input)
    denormalized = normalizer.denormalize(normalized)
    assert normalized == denormalized


if __name__ == "__main__":
    test_normalize_denormalize()
