"""
Global configuration object.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"

import json
from collections import namedtuple
from models import leakyrelu as LeakyReLUModel
from models import acapellabot as AcapellaBotModel
from models import unmix as UnmixModel
from exceptions.configurationerror import ConfigurationError
from helpers import converter


class Configuration(object):

    @staticmethod
    def load(configuration_file):
        global configuration
        with open(configuration_file, 'rb') as f:
            configuration = json.load(f, object_hook=lambda d: namedtuple(
                'X', d.keys())(*map(lambda x: converter.try_eval(x), d.values())))
        return configuration

    @staticmethod
    def get_model():
        model = configuration.training.model
        if model and model.name:
            if model.name.lower() == LeakyReLUModel.name.lower():
                return LeakyReLUModel.generate(model.alpha1, model.alpha2, model.rate)
            elif model.name.lower() == AcapellaBotModel.name.lower():
                return AcapellaBotModel.generate(0)
            elif model.name.lower() == UnmixModel.name:
                return UnmixModel.generate()
        raise ConfigurationError('training.model')
