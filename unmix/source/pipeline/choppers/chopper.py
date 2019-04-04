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

    # def get_chop(self, input, index, size):
    #     start = self.step * index - int(size / 2)
    #     end = start + size
    #     chop = transposer.pre_post_transpose(input, lambda x: self.pad_chop(x, start, end))
    #     return chop

    # def pad_chop(self, input, start, end):
    #     prepend = np.zeros((0, *input.shape[1:]))
    #     if start < 0:
    #         prepend = np.zeros((-start, *input.shape[1:]))
    #         end += -start
    #         start = 0
    #     append = np.zeros((0, *input.shape[1:]))
    #     if end >= len(input):
    #         append = np.zeros((end - len(input), *input.shape[1:]))
    #     return np.concatenate((prepend, input, append), axis=0)[start:end]

    def chop_n_pad(self, input, index, size):
        start = self.step * index - int(size / 2)
        end = start + size
        pad_count_left = abs(max(0, -start))
        pad_count_right = abs(max(0, end - input.shape[1]))
        chunk = input[:, max(0, start):min(end, input.shape[1])]
        return np.pad(chunk, ((0,0),(pad_count_left,pad_count_right)), "constant")

    def calculate_chops(self, input_width, size):
        max_width = input_width + int(size / 2) # at each end of the input, we maximal pad size / 2
        return int(math.ceil(max_width / self.step))
