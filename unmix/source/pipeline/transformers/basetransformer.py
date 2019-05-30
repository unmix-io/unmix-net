#!/usr/bin/env python3
# coding: utf8

"""
Base shared by all data transformers.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"


import numpy as np

from unmix.source.pipeline.choppers.chopper import Chopper
from unmix.source.configuration import Configuration
from unmix.source.helpers import spectrogramhandler
import unmix.source.pipeline.normalizers.normalizer_real_imag as normalizer_real_imag


class BaseTransformer(object):

    def __init__(self, size, step, shuffle):
        self.size = size
        self.shuffle = shuffle
        self.step = step
        self.chopper = Chopper(step)
        self.stereo = Configuration.get('collection.stereo', default=False)

    def calculate_items(self, width):
        return self.chopper.calculate_chops(width, self.size)

    def transform_input(self, data, index):
        input = [self.chopper.chop_n_pad(data[0], index, self.size)]
        if self.stereo:
            input.append(self.chopper.chop_n_pad(data[1], index, self.size))
        return input

    def prenormalize_window(self, mix, vocals, index):
        input = self.transform_input(mix, index)
        target = self.transform_input(vocals, index)

        normalized_input = normalizer_real_imag.normalize(input)
        normalized_target = normalizer_real_imag.normalize(target)

        # Reshape to put channels information in last dimension
        normalized_input = np.reshape(
            normalized_input, normalized_input.shape[1:-1] + (2 if self.stereo else 1,))
        normalized_target = np.reshape(
            normalized_target, normalized_target.shape[1:-1] + (2 if self.stereo else 1,))

        return normalized_input, normalized_target

    def prepare_window(self, mix, index):
        """
        Selects one training slice and performs preparation steps for the input (mix).
        """
        input = [self.chopper.chop_n_pad(mix[0], index, self.size)]
        if self.stereo:
            input.append(self.chopper.chop_n_pad(mix[1], index, self.size))
        normalized = normalizer_real_imag.normalize(input)
        normalized = np.reshape(
            normalized, normalized.shape[1:-1] + (2 if self.stereo else 1,))
        return normalized

    def save_audio(self, name, index, mix, vocals, normalized_input, normalized_target):
        spectrogramhandler.to_audio('%s-%d_Reconstructed_Input.wav' % (
            name, index), self.untransform_target(mix[0], normalized_input, index)[0])
        spectrogramhandler.to_audio('%s-%d_Reconstructed_Target.wav' % (
            name, index), self.untransform_target(mix[0], normalized_target, index)[0])

    def save_image(self, name, index, input, target):
        spectrogramhandler.to_image(
            '%s-%d_Target.png' % (name, index), target)
        spectrogramhandler.to_image(
            '%s-%d_Input.png' % (name, index), normalizer_real_imag.normalize(input))

    def untransform_target(self, *args):
        raise NotImplementedError(
            "Not definined by transformer implementation.")
