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
from unmix.source.configuration import Configuration


class TrainWindowPredictMaskTransformer:

    NAME = "train_window_predict_mask"

    def __init__(self, size, step, shuffle, save_audio):
        self.size = size
        self.shuffle = shuffle
        self.step = step
        self.save_audio = save_audio
        self.chopper = Chopper(step)

    def run(self, name, mix, vocals, index):
        input = self.chopper.chop_n_pad(mix[0], index, self.size) # we select input[0] here because we just use the left or mono channel for now
        target = self.chopper.chop_n_pad(vocals[0], index, self.size)
        
        normalized_input = normalizer_real_imag.normalize(input)[0]
        normalized_target = normalizer_real_imag.normalize(target)[0]

        max_input = np.max(normalized_input)
        if max_input > 0:
            normalized_input = normalized_input / max_input
            normalized_target = normalized_target / max_input

        if self.save_audio:
            spectrogramhandler.to_audio('%s-%d_Input.wav' % (name, index), input)
            spectrogramhandler.to_audio('%s-%d_Target.wav' % (name, index), target)

            spectrogramhandler.to_audio('%s-%d_Reconstructed_Input.wav' % (name, index), self.untransform_target(mix[0], normalized_input, index, (0,))[0])
            spectrogramhandler.to_audio('%s-%d_Reconstructed_Target.wav' % (name, index), self.untransform_target(mix[0], normalized_target, index, (0,))[0])

        return normalized_input, normalized_target

    def calculate_items(self, width):
        return self.chopper.calculate_chops(width, self.size)

    def prepare_input(self, mix, index):
        'Selects one training slice and performs preparation steps for the input (mix).'
        input = self.chopper.chop_n_pad(mix, index, self.size)

        normalized = normalizer_real_imag.normalize(input)
        normalized_data = normalized[0]

        data_max = np.max(normalized_data)
        if data_max > 0:
            normalized_data = normalized_data / np.max(normalized_data)

        return normalized_data, normalized[1]


    def untransform_target(self, mix, predicted_mask, index, transform_info):
        'Transforms predicted slices back to a format which corresponds to the training data (ready to process back to audio).'
        predicted_mask = np.reshape(predicted_mask, predicted_mask.shape[0:2])
        mix_slice = self.chopper.chop_n_pad(mix, index, self.step)
        mix_magnitude = np.abs(mix_slice)
        predicted_mask = np.clip(predicted_mask, 0, 1)
        vocal_magnitude = mix_magnitude * predicted_mask
        vocals = vocal_magnitude * np.exp(np.angle(mix_slice) * 1j)

        return vocals, mix_slice - vocals