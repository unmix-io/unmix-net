"""
Global configuration object.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"


import json
from collections import namedtuple

from helpers import converter
from helpers import reducer
from exceptions.configurationerror import ConfigurationError


class Configuration(object):

    @staticmethod
    def initialize(configuration_file):
        global configuration
        with open(configuration_file, 'rb') as f:
            configuration = json.load(f, object_hook=lambda d: namedtuple(
                'X', d.keys())(*map(lambda x: converter.try_eval(x), d.values())))
        return configuration_file

    @staticmethod
    def get(key='', optional=True):
        if not configuration:
            raise ConfigurationError()
        if key:
            try:
                value = reducer.rgetattr(configuration, key)
                if not optional and not value:
                    raise ConfigurationError(key)
                return value
            except:
                raise ConfigurationError(key)
        return key
