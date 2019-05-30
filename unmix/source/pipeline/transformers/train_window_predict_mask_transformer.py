#!/usr/bin/env python3
# coding: utf8

"""
Process the data using image representing windows for input and target.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"


import numpy as np

from unmix.source.pipeline.transformers.basetransformer import BaseTransformer
from unmix.source.exceptions.configurationerror import ConfigurationError
from unmix.source.helpers import reducer
from unmix.source.pipeline.choppers.chopper import Chopper
import unmix.source.pipeline.normalizers.normalizer_real_imag as normalizer_real_imag
import unmix.source.pipeline.normalizers.normalizer_max as normalizer_max
import unmix.source.pipeline.normalizers.normalizer_min_max as normalizer_min_max
from unmix.source.configuration import Configuration


class TrainWindowPredictMaskTransformer(BaseTransformer):

    NAME = "train_window_predict_mask"

    def __init__(self, size, step, shuffle, save_audio, normalizer_name):
        super().__init__(size, step, shuffle)
        self.save_audio = save_audio
        if normalizer_name == normalizer_max.name:
            self.normalizer = normalizer_max
        elif normalizer_name == normalizer_min_max.name:
            self.normalizer = normalizer_min_max
        else:
            self.normalizer = None

    def run(self, name, mix, vocals, index):
        normalized_input, normalized_target = super().prenormalize_window(
            mix, vocals, index)

        if self.normalizer:
            normalized_input, normalized_target = self.normalizer.normalize(
                normalized_input, normalized_target)

        if self.save_audio:
            super().save_audio(name, index, mix, vocals, normalized_input, normalized_target)
        return normalized_input, normalized_target

    def prepare_input(self, mix, index):
        normalized = super().prepare_window(mix, index)
        if self.normalizer:
            normalized, _ = self.normalizer.normalize(normalized)
        return normalized

    def untransform_target(self, mix, predicted_mask, index):
        'Transforms predicted slices back to a format which corresponds to the training data (ready to process back to audio).'
        mix_slice = [self.chopper.chop_n_pad(
            channel, index, self.size) for channel in mix]
        mix_magnitude = np.abs(mix_slice)

        predicted_mask = np.clip(predicted_mask, 0, 1)
        predicted_mask_reshape = [predicted_mask[:, :, 0]]
        if self.stereo:
            predicted_mask_reshape.append(predicted_mask[:, :, 1])

        vocal_magnitude = mix_magnitude * predicted_mask_reshape
        vocals = vocal_magnitude * np.exp(np.angle(mix_slice) * 1j)

        return vocals, mix_slice - vocals
