#!/usr/bin/env python3
# coding: utf8

"""
Fakes a chopper and does nothing with the input data.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"


import numpy as np

from unmix.source.exceptions.configurationerror import ConfigurationError
import unmix.source.helpers.transposer as transposer


class EmptyChopper:

    def __init__(self):
        self.mode = "None"
        self.inner_shape = None

    def chop(self, input):
        self.inner_shape = input.shape
        return np.array([input])

    def calculate_chops(self, width):
        return 1
