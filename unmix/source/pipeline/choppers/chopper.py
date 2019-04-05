#!/usr/bin/env python3
# coding: utf8

"""
Chops an input matrix horizontally.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"


import numpy as np
import math

from unmix.source.exceptions.configurationerror import ConfigurationError
import unmix.source.helpers.transposer as transposer


class Chopper:

    def __init__(self, step):
        self.step = step

    def chop_n_pad(self, input, index, size):
        start = self.step * index - int(size / 2)
        end = start + size
        pad_count_left = -min(0, start)
        pad_count_right = -min(0, input.shape[1] - end) + min(0, input.shape[1] - start)
        chunk = input[:, max(0, start):min(end, input.shape[1])]
        return np.pad(chunk, ((0,0),(pad_count_left,pad_count_right)), "constant")

    def calculate_chops(self, input_width, size):
        max_width = input_width + int(size / 2) # at each end of the input, we maximal pad size / 2
        return int(math.ceil(max_width / self.step))
