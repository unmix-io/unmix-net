#!/usr/bin/env python3
# coding: utf8

"""
Process the data using image representing windows for input and target.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"


import numpy as np

from unmix.source.exceptions.configurationerror import ConfigurationError
from unmix.source.helpers import spectrogramhandler
from unmix.source.helpers import reducer
from unmix.source.pipeline.choppers.chopper import Chopper
import unmix.source.pipeline.normalizers.normalizer_real_imag as normalizer_real_imag
import unmix.source.pipeline.normalizers.normalizer_zmuv as normalizer_zmuv
from unmix.source.configuration import Configuration


class WindowTransformer:

    NAME = "window"

    def __init__(self, size, step, shuffle, save_audio):
        self.size = size
        self.shuffle = shuffle
        self.save_audio = save_audio
        self.chopper = Chopper(step)

    def run(self, name, mix, vocals, index):
        input = self.chopper.chop_n_pad(mix[0], index, self.size) # we select input[0] here because we just use the left or mono channel for now
        target = self.chopper.chop_n_pad(vocals[0], index, self.size)
        
        if self.save_audio:
            spectrogramhandler.to_audio('%s-%d_Input.wav' % (name, index), input)
            spectrogramhandler.to_audio('%s-%d_Target.wav' % (name, index), target)

        normalized_input = normalizer_real_imag.normalize(input)[0]
        normalized_target = normalizer_real_imag.normalize(target)[0]
        
        mean = Configuration.get('collection.mean')
        variance = Configuration.get('collection.variance')
        if mean and variance:
            normalized_input = normalizer_zmuv.normalize(normalized_input, mean, variance)
            normalized_target = normalizer_zmuv.normalize(normalized_target, mean, variance)

        return normalized_input, normalized_target

    def calculate_items(self, width):
        return self.chopper.calculate_chops(width, self.size)

    def prepare_input(self, mix, index):
        'Selects one training slice and performs preparation steps for the input (mix).'
        input = self.chopper.chop_n_pad(mix, index, self.size)

        normalized = normalizer_real_imag.normalize(input)
        normalized_data = normalized[0]

        mean = Configuration.get('collection.mean')
        variance = Configuration.get('collection.variance')
        if mean and variance:
            normalized_data = normalizer_zmuv.normalize(normalized_data, mean, variance)

        return normalized_data, normalized[1]

    def untransform_target(self, mix, predicted, index, transform_info):
        'Transforms predicted slices back to a format which corresponds to the training data (ready to process back to audio).'
        mix_slice = self.chopper.chop_n_pad(mix, index, self.size)

        mean = Configuration.get('collection.mean')
        variance = Configuration.get('collection.variance')
        if mean and variance:
            predicted = normalizer_zmuv.denormalize(predicted, mean, variance)
        
        denormalized = normalizer_real_imag.denormalize(predicted, mix_slice, transform_info)
        return denormalized, mix_slice - denormalized