#!/usr/bin/env python3
# coding: utf8

"""
Normalizes data from maximum value.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"


import json
import numpy as np
import math

from unmix.source.configuration import Configuration
from unmix.source.exceptions.configurationerror import ConfigurationError
from unmix.source.helpers import filehelper


name = 'max'


def normalize(input, target):
    max = np.max(input)
    if max > 0:
        input = input / max
        target = target / max
    return input, target


def denormalize(input, target):
    return input, target
