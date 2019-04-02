#!/usr/bin/env python3
# coding: utf8

"""
Process the data using a probablistic filter mask.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"


import numpy as np

from unmix.source.exceptions.configurationerror import ConfigurationError
from unmix.source.helpers import audiohandler
from unmix.source.pipeline.choppers.chopper import Chopper


class MaskTransformer:

    NAME = "mask"

    def __init__(self, window, step, shuffle):
        self.window = window
        self.shuffle = shuffle
        self.chopper = Chopper(step)

    def run(self, name, mix, vocals, index):
        x = []
        y = []
        return x, y

    def calculate_items(self, width):
        return self.chopper.calculate_chops(width)
