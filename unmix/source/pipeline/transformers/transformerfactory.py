#!/usr/bin/env python3
# coding: utf8

"""
Builds a transformer for processing the data.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"


import keras.backend as keras

from unmix.source.configuration import Configuration
from unmix.source.exceptions.configurationerror import ConfigurationError
from unmix.source.pipeline.transformers.masktransformer import MaskTransformer
from unmix.source.pipeline.transformers.mask_ibm_transformer import IBMMaskTransformer
from unmix.source.pipeline.transformers.windowtransformer import WindowTransformer
from unmix.source.pipeline.transformers.train_window_predict_mask_transformer import TrainWindowPredictMaskTransformer


class TransformerFactory(object):

    @staticmethod
    def build():
        name = Configuration.get('transformation.name', optional=False)
        options = Configuration.get('transformation.options', optional=False)
        try:
            if name == WindowTransformer.NAME:
                return WindowTransformer(options.size, options.step, options.shuffle, options.save_audio)
            if name == MaskTransformer.NAME:
                return MaskTransformer(options.size, options.step, options.shuffle, options.save_image)
            if name == IBMMaskTransformer.NAME:
                return IBMMaskTransformer(options.size, options.step, options.shuffle, options.save_image)
            if name == TrainWindowPredictMaskTransformer.NAME:
                return TrainWindowPredictMaskTransformer(options.size, options.step, options.shuffle, options.save_audio)
        except:
            raise ConfigurationError('transformation.options')
        raise ConfigurationError('transformation.name')
