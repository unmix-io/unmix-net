#!/usr/bin/env python3
# coding: utf8

"""
Transform input to normalized amplitude spectrograms, target to a probability mask for vocals
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"


import numpy as np

from unmix.source.pipeline.transformers.basetransformer import BaseTransformer
from unmix.source.configuration import Configuration
from unmix.source.pipeline.choppers.chopper import Chopper
import unmix.source.pipeline.normalizers.normalizer_real_imag as normalizer_real_imag
from unmix.source.helpers.masker import mask
from unmix.source.helpers import spectrogramhandler

class MaskTransformer(BaseTransformer):

    NAME = "mask"

    def __init__(self, size, step, shuffle, save_image):
        super().__init__(size, step, shuffle)
        self.save_image = save_image

    def run(self, name, mix, vocals, index):
        """
        Returns: (769,size,1), (769,step,1)
        """
        mix_slice = super().transform_input(mix, index)
        vocal_slice = super().transform_input(mix, index)
        
        mix_magnitude = np.abs(mix_slice)
        vocal_magnitude = np.abs(vocal_slice)

        target_mask = mask(vocal_magnitude, mix_magnitude)

        if self.save_image:
            super().save_image(name, index, mix_slice, target_mask)
            
        target_mask = np.reshape(target_mask, target_mask.shape + (1,))
        return normalizer_real_imag.normalize(input), target_mask

    def prepare_input(self, mix, index):
        """
        Selects one training slice and performs preparation steps for the input (mix).
        """
        input = super().transform_input(mix, index)
        return normalizer_real_imag.normalize(input)

    def untransform_target(self, mix, predicted_mask, index):
        'Transforms predicted slices back to a format which corresponds to the training data (ready to process back to audio).'
        predicted_mask = np.reshape(predicted_mask, predicted_mask.shape[0:2])
        mix_slice = self.chopper.chop_n_pad(mix, index, self.step)
        mix_magnitude = np.abs(mix_slice)
        predicted_mask = np.clip(predicted_mask, 0, 1)
        vocal_magnitude = mix_magnitude * predicted_mask
        vocals = vocal_magnitude * np.exp(np.angle(mix_slice) * 1j)

        return vocals, mix_slice - vocals