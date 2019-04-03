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
from unmix.source.pipeline.transformers.windowtransformer import WindowTransformer


class TransformerFactory(object):

    @staticmethod
    def build():
        name = Configuration.get('transformation.name', False)
        options = Configuration.get('transformation.options', False)
        try:
            if name == MaskTransformer.NAME:
                return MaskTransformer(options.size, options.step, options.shuffle)
            if name == WindowTransformer.NAME:
                return WindowTransformer(options.size, options.step, options.shuffle, options.save_audio)
        except:
            raise ConfigurationError('transformation.options')
        raise ConfigurationError('transformation.name')
