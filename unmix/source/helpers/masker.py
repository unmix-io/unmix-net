#!/usr/bin/env python3
# coding: utf8

"""
Create relational masks of matrices.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"


import numpy as np


def mask(a, b, position=0):
    return np.clip(a / b, 0, 1)
