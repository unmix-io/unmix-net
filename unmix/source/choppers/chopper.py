#!/usr/bin/env python3
# coding: utf8

"""
Builds choppers to chop the input spectrograms.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"


import numpy as np

from unmix.source.exceptions.configurationerror import ConfigurationError


class Chopper:

    DIRECTION_HORIZONTAL = "horizontal"
    DIRECTION_VERTICAL = "vertical"
    MODE_SPLIT = "split"
    MODE_OVERLAP = "overlap"

    PRE_TRANSPOSE_DIMENSIONS = [[0], [1,0], [1,0,2], [2,0,1,3]]
    POST_TRANSPOSE_DIMENSIONS = [[0,1], [0,2,1], [0,2,1,3], [0,2,1,3,4]]

    def __init__(self, direction, mode, size):
        self.direction = direction
        self.mode = mode
        self.size = size

    def chop(self, input):
        if self.direction == Chopper.DIRECTION_VERTICAL:
            if self.mode == Chopper.MODE_SPLIT:
                return self.chop_split(input)
            if self.mode == Chopper.MODE_OVERLAP:
                return self.chop_overlap(input)
        if self.direction == Chopper.DIRECTION_HORIZONTAL:
            if self.mode == Chopper.MODE_SPLIT:
                return self.chop_split(input.transpose(*Chopper.PRE_TRANSPOSE_DIMENSIONS[len(input.shape) - 1])) \
                        .transpose(*Chopper.POST_TRANSPOSE_DIMENSIONS[len(input.shape) - 1])
            if self.mode == Chopper.MODE_OVERLAP:
                return self.chop_overlap(input.transpose(*Chopper.PRE_TRANSPOSE_DIMENSIONS[len(input.shape) - 1])) \
                        .transpose(*Chopper.POST_TRANSPOSE_DIMENSIONS[len(input.shape) - 1])
        raise ConfigurationError("Chopper with invalid configuration")

    def chop_split(self, input):
        slices = int(len(input) / self.size)
        chops = [input[(i * self.size):((i+1) * self.size)] for i in range(slices)]
        return np.array(chops)

    def chop_overlap(self, input):
        chops = []
        position = 0
        step = int(self.size / 2)
        while position + step < len(input):
            chops.append(input[position:(position + self.size)])
            position += step
        return np.array(chops)

    def calculate_chops(self, width, height):
        chops = 0
        if self.direction == Chopper.DIRECTION_VERTICAL:
            chops = height / self.size
        if self.direction == Chopper.DIRECTION_HORIZONTAL:
            chops = width / self.size
        if self.mode == Chopper.MODE_OVERLAP:
            chops = chops * 2 - 1
        return int(chops)