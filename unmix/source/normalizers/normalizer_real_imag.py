#!/usr/bin/env python3
# coding: utf8

"""
Normalizes real and imagninary matrix values, used for the leakyrelu model.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"


import numpy as np
import math

name = 'normalizer_real_imag'

def normalize(realimag):
    'Normalizes real and imaginary part'

    realimag = realimag / np.percentile(np.abs(realimag), 99) # normalize to the 99th percentile
    realimag = np.clip(realimag, -1, 1)
    return realimag

def denormalize(magnitude_angle):
    'Returns denormalized values as array of complex values'
    return magnitude_angle
