#!/usr/bin/env python3
# coding: utf8

"""
Tests the track mixing.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"


import numpy as np
from functools import reduce


def test_mix_reduce():
    a = np.array([0, 1, 2])
    b = np.array([1, 2, 3])
    c = np.array([2, 3, 4])

    mix = reduce((lambda x, y: x + y), [a, b,c])
    assert len(mix) == 8


if __name__ == "__main__":
    test_mix_reduce()
    print("Test run successful.")
