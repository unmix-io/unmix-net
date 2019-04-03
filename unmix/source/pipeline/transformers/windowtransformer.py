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
from unmix.source.helpers import reducer
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
        input = reducer.rflatter(self.chopper.get_chop(mix, index, self.size)) # todo: without save_audio, this could use prepare_input below
        target = reducer.rflatter(self.chopper.get_chop(vocals, index, self.size))

        if self.save_audio:
            audiohandler.spectrogram_to_audio('%s_mix.wav' % name, input)
            audiohandler.spectrogram_to_audio('%s_vocals.wav' % name, target)
        
        return normalizer.normalize(input)[0], normalizer.normalize(target)[0]

    def calculate_items(self, width):
        return self.chopper.calculate_chops(width)

    def prepare_input(self, mix, index):
        'Selects one training slice and performs preparation steps for the input (mix).'
        input = self.chopper.get_chop(mix, index, self.size)
        return normalizer.normalize(input)

    def untransform_target(self, mix, predicted, index, transform_info):
        'Transforms predicted slices back to a format which corresponds to the training data (ready to process back to audio).'
        mix_slice = self.chopper.get_chop(mix, index, self.size)
        return normalizer.denormalize(predicted, mix_slice, transform_info)