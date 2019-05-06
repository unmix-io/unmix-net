#!/usr/bin/env python3
# coding: utf8

"""
Normalizes data from minimum and maximum value.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"


import json
import numpy as np
import math

from unmix.source.configuration import Configuration
from unmix.source.exceptions.configurationerror import ConfigurationError
from unmix.source.helpers import filehelper


name = 'min_max'


def normalize(input, target):
    max = np.max(input)
    min = np.min(input)
    difference = max - min
    if difference > 0:
        input = (input - min) / difference
        target = (input - min) / difference
    return input, target


def denormalize(input, target):
    return input, target
