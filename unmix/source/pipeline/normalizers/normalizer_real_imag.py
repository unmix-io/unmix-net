#!/usr/bin/env python3
# coding: utf8

"""
Normalizes real and imagninary matrix values, used for the leakyrelu model.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"


import numpy as np
import math

name = 'norm_real_imag'

def normalize(track_complex):
    """
    Normalizes training data to use only amplitudes
    """
    magnitude = np.abs(track_complex)
    magnitude = np.reshape(magnitude, magnitude.shape + (1,))
    return magnitude

def denormalize(magnitude_predicted, mix_complex):
    """
    Returns denormalized values as array of complex values (sft)
    """

    denorm = np.reshape(magnitude_predicted, magnitude_predicted.shape[0:2])       
    denorm = denorm.clip(0)

    denorm = denorm * np.exp(np.angle(mix_complex) * 1j)
    return denorm
