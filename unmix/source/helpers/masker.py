#!/usr/bin/env python3
# coding: utf8

"""
Create relational masks of matrices.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"


import numpy as np


def mask(a, b):
    return np.clip(np.divide(a, b, out=np.zeros_like(a), where=b != 0), 0, 1)
