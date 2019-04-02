#!/usr/bin/env python3
# coding: utf8

"""
Chops an input matrix horizontally.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"


import numpy as np

from unmix.source.exceptions.configurationerror import ConfigurationError
import unmix.source.helpers.transposer as transposer


class Chopper:

    def __init__(self, step):
        self.step = step

    def get_chop(self, input, index, size):
        start = self.step * index - int(size / 2)
        end = start + size
        chop = transposer.pre_post_transpose(input, lambda x: self.pad_chop(x, start, end))
        return chop

    def pad_chop(self, input, start, end):
        prepend = np.zeros((0, *input.shape[1:]))
        if start < 0:
            prepend = np.zeros((-start, *input.shape[1:]))
            end += -start
            start = 0
        append = np.zeros((0, *input.shape[1:]))
        if end >= len(input):
            append = np.zeros((end - len(input), *input.shape[1:]))
        return np.concatenate((prepend, input, append), axis=0)[start:end]

    def calculate_chops(self, width):
        return int(width / self.step)
