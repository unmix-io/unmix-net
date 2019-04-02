#!/usr/bin/env python3
# coding: utf8

"""
Process the data using a probablistic filter mask.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"


import numpy as np

from unmix.source.exceptions.configurationerror import ConfigurationError
import unmix.source.helpers.transposer as transposer


class MaskTransformer:

    NAME = "mask"

    def __init__(self, window, step, shuffle, save_audio):
        self.step = window
        self.step = step
        self.step = shuffle
        self.step = save_audio

    def run(self, mix, vocals, index):
        x = []
        y = []
        return x, y
