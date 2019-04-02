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
        return input[start:end] # TODO: Pad zeros if outside boundaries

    def calculate_chops(self, width):
        return int(width / self.step)
