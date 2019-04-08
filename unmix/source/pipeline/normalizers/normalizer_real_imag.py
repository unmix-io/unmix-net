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

def normalize(cplx):
    'Normalizes training data to use only amplitudes'

    magnitude = np.abs(cplx)
    percentile99 = np.percentile(magnitude, 99)
    #magnitude = np.clip(magnitude, 0, percentile99)

    #if(percentile99 > 0):
    #    magnitude = magnitude / (percentile99 / 2)
    
    #magnitude = magnitude - 1
    magnitude = np.reshape(magnitude, magnitude.shape + (1,))
    return magnitude, (percentile99,)

def denormalize(magnitude_predicted, mix_complex, normalizer_info):
    'Returns denormalized values as array of complex values (sft)'
    percentile99 = normalizer_info[0]
    denorm = np.reshape(magnitude_predicted, magnitude_predicted.shape[0:2])
    
    # Shift values to original space and clip values below zero
    #denorm = denorm + 1
    
    denorm = denorm.clip(0)
    #denorm = denorm * (percentile99 / 2)

    denorm = denorm * np.exp( np.angle(mix_complex) * 1j )
    return denorm
