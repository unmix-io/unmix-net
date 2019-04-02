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

    def __init__(self, step, size, offset, save_audio):
        self.step = step
        self.size = size
        self.offset = offset
        self.save_audio = save_audio
        self.inner_shape = None

    def get_chop(self, input, offset):
        start = 0 if self.padding else self.offset
        end = len()

    def set_inner_shape(self, chops):
        if chops.any():
            self.inner_shape = chops[0].shape

    def calculate_chops(self, width):
        return int(width / self.step)
