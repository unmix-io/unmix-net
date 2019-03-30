#!/usr/bin/env python
# coding: utf8

"""
Normalizes real and imagninary matrix values.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"


import numpy as np
import math


def polar_to_rectangular(radii, angles):
    return radii * np.exp(1j*angles)

def normalize(realimag):
    'Transforms real and imaginary parts to magnitude and angle (combined into array); scaling it between -1 and 1'

    real = realimag[:, :, 0]
    imag = realimag[:, :, 1]
    cplx = real + imag * 1j

    magnitude = np.abs(cplx)
    angle = np.angle(cplx)

    angle = angle / math.pi # angle is always between -pi and +pi - scale from -1 to 1

    # TODO: Test if log scale performs better
    # magnitude = np.log1p(magnitude)
    #max_magnitude = np.max(np.abs(magnitude))

    #if max_magnitude != 0:
    #    magnitude = magnitude / (max_magnitude / 2) - 1 # scale from -1 to 1
    magnitude = magnitude / 769 # TODO: get from parameters

    combined = np.empty(realimag.shape, dtype=realimag.dtype)
    combined[:, :, 0] = magnitude
    combined[:, :, 1] = angle
    return combined

def denormalize(magnitude_angle):
    'Returns denormalized values as array of complex values'
    magnitude = magnitude_angle[:, :, 0]
    magnitude = magnitude * 769 # TODO: get from parameters
    angle = magnitude_angle[:, :, 1]
    angle = angle * math.pi
    cplx = polar_to_rectangular(magnitude, angle)
    return cplx
