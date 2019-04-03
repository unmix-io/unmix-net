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

    def __init__(self, size, step, shuffle, save_audio):
        self.size = size
        self.shuffle = shuffle
        self.save_audio = save_audio
        self.chopper = Chopper(step)

    def run(self, name, mix, vocals, index):
        input = reducer.rflatter(self.chopper.get_chop(mix, index, self.size))

        # Calculate mask
        mask_index = self.get_mask_index(index)
        mix_slice = reducer.rflatter(self.chopper.get_chop(mix, mask_index, self.size))
        vocal_slice = reducer.rflatter(self.chopper.get_chop(vocals, mask_index, self.size))
        target_mask = mask(vocal_slice, mix_slice)

        return normalizer.normalize(input)[0], target_mask

    def calculate_items(self, width):
        return self.chopper.calculate_chops(width)

    def prepare_input(self, mix, index):
        'Selects one training slice and performs preparation steps for the input (mix).'
        input = self.chopper.get_chop(mix, index, self.size)
        return normalizer.normalize(input)

    def untransform_target(self, mix, predicted_mask, index, transform_info):
        'Transforms predicted slices back to a format which corresponds to the training data (ready to process back to audio).'
        mix_slice = self.chopper.get_chop(mix, self.get_mask_index(index), self.size)
        return mix_slice * predicted_mask

    def get_mask_index(self, index):
         return index + int(self.size/2)