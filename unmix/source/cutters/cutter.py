#!/usr/bin/env python3
# coding: utf8

"""
Cuts out data from a matrix.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"


import numpy as np

from unmix.source.exceptions.configurationerror import ConfigurationError
import unmix.source.helpers.transposer as transposer


class Cutter:

    POSITION_LEFT = "left"
    POSITION_CENTER = "center"
    POSITION_RIGHT = "right"

    def __init__(self, position, size):
        self.position = position
        self.size = size

    def cut(self, input, transpose=True):
        if transpose:
            input = transposer.pre_transpose(input)
        if self.position == Cutter.POSITION_LEFT:
            output = input[:self.size]
        if self.position == Cutter.POSITION_CENTER:
            position = int(len(input)/2 - self.size/2)
            output = input[position:position+self.size]
        if self.position == Cutter.POSITION_RIGHT:
            output = input[-self.size:]
        if transpose:
            output = transposer.post_transpose(output)
        return output
