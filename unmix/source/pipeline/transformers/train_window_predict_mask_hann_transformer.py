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
from unmix.source.pipeline.transformers.train_window_predict_mask_transformer import TrainWindowPredictMaskTransformer

class TrainWindowPredictMaskHannTransformer(TrainWindowPredictMaskTransformer):

    NAME = "train_window_predict_mask_hann"

    def __init__(self, size, step, shuffle, save_audio, normalizer_name):
        if(size % step != 0):
            raise Exception("Step must be a multiple of the size.")
        super().__init__(size, step, shuffle, save_audio, normalizer_name)

    def __hann(self, matrix):
        for channel in range(matrix.shape[-1]):
            matrix[:,:,channel] = matrix[:,:,channel] * np.hanning(self.size)
        return matrix

    def run(self, name, mix, vocals, index):
        normalized_input, normalized_target = super().run(name, mix, vocals, index)

        #normalized_input = self.__hann(normalized_input)
        #normalized_target = self.__hann(normalized_target)

        return normalized_input, normalized_target

    def prepare_input(self, mix, index):
        normalized = super().prepare_input(mix, index)

        #normalized = self.__hann(normalized)

        return normalized

    def untransform_target(self, mix, predicted_mask, index):
        'Transforms predicted slices back to a format which corresponds to the training data (ready to process back to audio).'
        vocals, instrumentals = super().untransform_target(mix, predicted_mask, index)

        #vocals = vocals * np.hanning(self.size)
        #instrumentals = instrumentals * np.hanning(self.size)

        return vocals, instrumentals

    def expand_track(self, prediction, track, i):
        'Append new predictions to the already made predictions matrix'
        left = self.step * i
        right = left + self.size
        size = track.shape[2]
        if size < right:
            track = np.append(track, np.zeros((track.shape[0], track.shape[1], right - size)), axis=2)
        # Sum up over the "hanned" predictions
        factor = self.size / self.step / 2
        track[:,:,left:right] += prediction * np.hanning(self.size) / factor

