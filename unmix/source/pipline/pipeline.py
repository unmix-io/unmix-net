#!/usr/bin/env python3
# coding: utf8

"""
Chops an input matrix horizontally.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"


import numpy as np

from unmix.source.pipline.choppers.chopper import Chopper
from unmix.source.exceptions.configurationerror import ConfigurationError
import unmix.source.helpers.transposer as transposer


class Pipeline:

    def __init__(self, type, chop, transform):
        self.type = type
        self.chopper = Chopper(chop.step, chop.size, chop.offset, chop.save_audio)
        self.transformer = None


    