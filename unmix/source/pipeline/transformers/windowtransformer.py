#!/usr/bin/env python3
# coding: utf8

"""
Process the data using image representing windows for input and target.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"


import numpy as np

from unmix.source.exceptions.configurationerror import ConfigurationError
from unmix.source.helpers import audiohandler
from unmix.source.pipeline.choppers.chopper import Chopper
import unmix.source.pipeline.normalizers.normalizer_real_imag as normalizer


class WindowTransformer:

    NAME = "window"

    def __init__(self, size, step, shuffle, save_audio):
        self.size = size
        self.shuffle = shuffle
        self.save_audio = save_audio
        self.chopper = Chopper(step)

    def run(self, name, mix, vocals, index):

        input = self.chopper.get_chop(vocals, index, self.size)
        target = self.chopper.get_chop(mix, index, self.size)

        if self.save_audio:
            audiohandler.spectrogram_to_audio(
                '%s-%s_mix.wav' % name, input)
            audiohandler.spectrogram_to_audio(
                '%s_vocals.wav' % name, target)

        return normalizer.normalize(input), normalizer.normalize(target)

    def calculate_items(self, width):
        return self.chopper.calculate_chops(width)
