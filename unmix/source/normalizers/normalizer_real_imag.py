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
    'Normalizes training data to use only amplitudes'
    real = realimag[:, :, 0]
    imag = realimag[:, :, 1]
    cplx = real + imag * 1j

    magnitude = np.abs(cplx)
    percentile99 = np.percentile(magnitude, 99)
    magnitude = np.clip(magnitude, 0, percentile99)

    magnitude = magnitude / (percentile99 * 2)
    magnitude = magnitude - 1

    return np.reshape(magnitude, magnitude.shape + (1,))

def denormalize(realimag):
    'Returns denormalized values as array of complex values'
    realimag = realimag * 1000 # Todo: value should be taken from normalizer
    realimag = realimag[:, :, 0] + realimag[:, :, 1] * 1j
    return realimag
