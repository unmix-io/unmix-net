#!/usr/bin/env python3
# coding: utf8

"""
Transform input to normalized amplitude spectrograms, target to a probability mask for vocals
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"


import numpy as np

from unmix.source.configuration import Configuration
from unmix.source.pipeline.choppers.chopper import Chopper
import unmix.source.pipeline.normalizers.normalizer_real_imag as normalizer
from unmix.source.helpers.masker import mask
from unmix.source.helpers import spectrogramhandler

class MaskTransformer:

    NAME = "mask"

    def __init__(self, size, step, shuffle, save_image):
        self.size = size
        self.step = step
        self.shuffle = shuffle
        self.save_image = save_image
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

        if self.save_image:
            spectrogramhandler.to_image('%s-%d_Target.png' % (name, index), target_mask)
            spectrogramhandler.to_image('%s-%d_Input.png' % (name, index), normalizer.normalize(input))
            spectrogramhandler.to_audio('%s-%d_Reconstructed.wav' % (name, index), mix_magnitude * target_mask * np.exp(np.angle(mix_slice) * 1j ))
            
        target_mask = np.reshape(target_mask, target_mask.shape + (1,))
        return normalizer.normalize(input), target_mask

    def calculate_items(self, width):
        return self.chopper.calculate_chops(width, self.size)

    def prepare_input(self, mix, index):
        'Selects one training slice and performs preparation steps for the input (mix).'
        input = self.chopper.chop_n_pad(mix, index, self.size)
        return normalizer.normalize(input)

    def untransform_target(self, mix, predicted_mask, index):
        'Transforms predicted slices back to a format which corresponds to the training data (ready to process back to audio).'
        predicted_mask = np.reshape(predicted_mask, predicted_mask.shape[0:2])
        mix_slice = self.chopper.chop_n_pad(mix, index, self.step)
        mix_magnitude = np.abs(mix_slice)
        predicted_mask = np.clip(predicted_mask, 0, 1)
        vocal_magnitude = mix_magnitude * predicted_mask
        vocals = vocal_magnitude * np.exp(np.angle(mix_slice) * 1j)

        return vocals, mix_slice - vocals