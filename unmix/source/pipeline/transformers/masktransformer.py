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
        input = reducer.rflatter(self.chopper.get_chop(mix, index, self.size))

        # Calculate mask
        mask_index = self.__get_mask_index(index)
        mix_slice = reducer.rflatter(self.chopper.get_chop(mix, mask_index, self.step))
        mix_magnitude = np.abs(self.__get_complex(mix_slice))
        vocal_slice = reducer.rflatter(self.chopper.get_chop(vocals, mask_index, self.step))
        vocal_magnitude = np.abs(self.__get_complex(vocal_slice))
        target_mask = mask(vocal_magnitude, mix_magnitude)
        target_mask = np.reshape(target_mask, target_mask.shape + (1,))

        return normalizer.normalize(input)[0], target_mask

    def calculate_items(self, width):
        return self.chopper.calculate_chops(width)

    def prepare_input(self, mix, index):
        'Selects one training slice and performs preparation steps for the input (mix).'
        input = self.chopper.get_chop(mix, index, self.size)
        return normalizer.normalize(input)

    def untransform_target(self, mix, predicted_mask, index, transform_info):
        'Transforms predicted slices back to a format which corresponds to the training data (ready to process back to audio).'
        mix_slice = self.chopper.get_chop(mix, self.__get_mask_index(index), self.step)
        mix_complex = self.__get_complex(mix_slice)
        mix_magnitude = np.abs(mix_complex)
        vocal_magnitude = mix_magnitude * predicted_mask
        vocals = vocal_magnitude * np.exp( np.angle(mix_complex) * 1j )

        return vocals

    def __get_mask_index(self, index):
         return index + int(self.size/2)

    def __get_complex(self, realimag):
        real = realimag[:, :, 0]
        imag = realimag[:, :, 1]
        return real + imag * 1j
