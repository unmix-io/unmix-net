#!/usr/bin/env python3
# coding: utf8

"""
Builds a normalizer from configuration.
"""

from unmix.source.configuration import Configuration
from unmix.source.exceptions.configurationerror import ConfigurationError
from unmix.source.normalizers import normalizer_abs
from unmix.source.normalizers import normalizer_real_imag

normalizers = [ normalizer_abs, normalizer_real_imag ]

class NormalizerFactory(object):

    @staticmethod
    def build():
        normalizer = Configuration.get('training.normalizer', False)
        if normalizer.name:
            return [n for n in normalizers if n.name == normalizer.name][0]
        raise ConfigurationError('training.model.name')
