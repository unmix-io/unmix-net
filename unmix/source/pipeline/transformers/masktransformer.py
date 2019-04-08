#!/usr/bin/env python3
# coding: utf8

"""
Transform input to normalized amplitude spectrograms, target to a probability mask for vocals
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"


import numpy as np

from unmix.source.exceptions.configurationerror import ConfigurationError
from unmix.source.helpers import audiohandler
from unmix.source.helpers import reducer
from unmix.source.pipeline.choppers.chopper import Chopper
import unmix.source.pipeline.normalizers.normalizer_real_imag as normalizer
from unmix.source.helpers.masker import mask

class MaskTransformer:

    NAME = "mask"

    def __init__(self, size, step, shuffle):
        self.size = size
        self.step = step
        self.shuffle = shuffle
        self.chopper = Chopper(step)

    def run(self, name, mix, vocals, index):
        'Returns shapes (769,size,1), (769,step,1)'
        input = self.chopper.chop_n_pad(mix[0], index, self.size)

        # Calculate mask
        mix_slice = self.chopper.chop_n_pad(mix[0], index, self.step)
        mix_magnitude = np.abs(mix_slice)
        vocal_slice = self.chopper.chop_n_pad(vocals[0], index, self.step)
        vocal_magnitude = np.abs(vocal_slice)
        target_mask = mask(vocal_magnitude, mix_magnitude)
        target_mask = np.reshape(target_mask, target_mask.shape + (1,))

        return normalizer.normalize(input)[0], target_mask

    def calculate_items(self, width):
        return self.chopper.calculate_chops(width, self.size)

    def prepare_input(self, mix, index):
        'Selects one training slice and performs preparation steps for the input (mix).'
        input = self.chopper.chop_n_pad(mix, index, self.size)
        return normalizer.normalize(input)

    def untransform_target(self, mix, predicted_mask, index, transform_info):
        'Transforms predicted slices back to a format which corresponds to the training data (ready to process back to audio).'
        predicted_mask = np.reshape(predicted_mask, predicted_mask.shape[0:2])
        mix_slice = self.chopper.chop_n_pad(mix, index, self.step)
        mix_magnitude = np.abs(mix_slice)
        vocal_magnitude = mix_magnitude * predicted_mask
        vocals = vocal_magnitude * np.exp( np.angle(mix_slice) * 1j )

        return vocals