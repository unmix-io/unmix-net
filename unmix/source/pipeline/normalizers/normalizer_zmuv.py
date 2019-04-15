#!/usr/bin/env python3
# coding: utf8

"""
Normalizes data to zero mean and unit variance.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"


import numpy as np
import math

name = 'normalizer_zmuv'

def normalize(data, mean, variance):
    'Normalizes training data to zero mean and unit variance.'
    return (data - mean) / variance

def denormalize(data, mean, variance):
    'Returns denormalized values aways with previous mean and variance.'    
    return (data * variance) + mean
