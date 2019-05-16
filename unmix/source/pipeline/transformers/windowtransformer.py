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
from unmix.source.helpers import spectrogramhandler
from unmix.source.helpers import reducer
from unmix.source.pipeline.choppers.chopper import Chopper
import unmix.source.pipeline.normalizers.normalizer_zmuv as normalizer_zmuv
from unmix.source.configuration import Configuration
import unmix.source.pipeline.normalizers.normalizer_real_imag as normalizer_real_imag


class WindowTransformer(BaseTransformer):

    NAME = "window"

    def __init__(self, size, step, shuffle, save_audio):
        super().__init__(size, step, shuffle)
        self.save_audio = save_audio

    def run(self, name, mix, vocals, index):
        normalized_input, normalized_target = super().prenormalize_window(mix, vocals, index)
        zmuv_normalizer_config = Configuration.get(
            'transformation.normalizers.zmuv')
        if zmuv_normalizer_config and zmuv_normalizer_config.enabled:
            normalized_input = normalizer_zmuv.normalize(
                normalized_input, zmuv_normalizer_config.mode, zmuv_normalizer_config.mix_file)
            normalized_target = normalizer_zmuv.normalize(
                normalized_target, zmuv_normalizer_config.mode, zmuv_normalizer_config.vocals_file)

        if self.save_audio:
            super().save_audio(name, index, mix, vocals,
                             normalized_input, normalized_target)
        return normalized_input, normalized_target

    def prepare_input(self, mix, index):
        normalized = super().prepare_window(mix, index)
        zmuv_normalizer_config = Configuration.get(
            'transformation.normalizers.zmuv')
        if zmuv_normalizer_config and zmuv_normalizer_config.enabled:
            normalized = normalizer_zmuv.normalize(
                normalized, zmuv_normalizer_config.mode, zmuv_normalizer_config.vocals_file)
        return normalized

    def untransform_target(self, mix, predicted, index):
        'Transforms predicted slices back to a format which corresponds to the training data (ready to process back to audio).'
        mix_slice = [self.chopper.chop_n_pad(
            channel, index, self.step) for channel in mix]

        zmuv_normalizer_config = Configuration.get(
            'transformation.normalizers.zmuv')
        if zmuv_normalizer_config and zmuv_normalizer_config.enabled:
            predicted = normalizer_zmuv.denormalize(
                predicted, zmuv_normalizer_config.mode, zmuv_normalizer_config.vocals_file)

        denormalized = normalizer_real_imag.denormalize(predicted, mix_slice)
        return denormalized, mix_slice - denormalized
