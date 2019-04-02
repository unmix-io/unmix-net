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

    DIRECTION_HORIZONTAL = "horizontal"
    MODE_SPLIT = "split"
    MODE_OVERLAP = "overlap"
    MODE_STEPWISE = "stepwise"

    def __init__(self, direction, mode, size):
        self.direction = direction
        self.mode = mode
        self.size = size
        self.inner_shape = None

    def chop(self, input):
        if self.direction == Chopper.DIRECTION_HORIZONTAL:
            if self.mode == Chopper.MODE_SPLIT:
                return transposer.post_transpose(self.chop_split(transposer.pre_transpose(input)))
            if self.mode == Chopper.MODE_OVERLAP:
                return transposer.post_transpose(self.chop_overlap(transposer.pre_transpose(input)))
            if self.mode == Chopper.MODE_STEPWISE:
                return transposer.post_transpose(self.chop_stepwise(transposer.pre_transpose(input)))
        raise ConfigurationError("Chopper with invalid configuration")

    def chop_split(self, input):
        slices = int(len(input) / self.size)
        chops = [input[(i * self.size):((i+1) * self.size)]
                 for i in range(slices)]
        self.set_inner_shape(chops)
        return np.array(chops)

    def chop_overlap(self, input):
        step = int(self.size / 2)
        return self.chop_step(input, step)

    def chop_stepwise(self, input):
        step = 1
        return self.chop_step(input, step)

    def chop_step(self, input, step):
        count = self.calculate_chops(len(input))
        chops = np.empty((count,) + (self.size,) + input.shape[1:])
        position = 0
        i = 0
        while i < count:
            chops[i] = input[position:(position + self.size)]
            position += step
            i += 1
        self.set_inner_shape(chops)
        return chops

    def set_inner_shape(self, chops):
        if chops.any():
            self.inner_shape = chops[0].shape

    def calculate_chops(self, width):
        chops = 0
        if self.mode == Chopper.MODE_STEPWISE:
            if self.direction == Chopper.DIRECTION_HORIZONTAL:
                chops = width - self.size + 1
        if self.mode == Chopper.MODE_OVERLAP:
            if self.direction == Chopper.DIRECTION_HORIZONTAL:
                chops = int((width - self.size / 2) / (self.size / 2))
        if self.mode == Chopper.MODE_SPLIT:
            if self.direction == Chopper.DIRECTION_HORIZONTAL:
                chops = int(width / self.size)
        return int(chops)
