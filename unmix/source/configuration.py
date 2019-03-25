"""
Global configuration object.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"


import os
import json
from collections import namedtuple

from unmix.source.helpers import reducer
from unmix.source.helpers import converter
from unmix.source.exceptions.configurationerror import ConfigurationError


class Configuration(object):

    working_directory = ''

    @staticmethod
    def initialize(configuration_file, working_directory=None):
        global configuration
        with open(configuration_file, 'rb') as f:
            configuration = json.load(f, object_hook=lambda d: namedtuple(
                'X', d.keys())(*map(lambda x: converter.try_eval(x), d.values())))
        Configuration.working_directory = working_directory if working_directory else os.getcwd()
        return configuration

    @staticmethod
    def get(key='', optional=True):
        global configuration
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

    @staticmethod
    def get_path(key='', optional=True):
        path = Configuration.get(key, optional)
        return Configuration.build_path(path)

    @staticmethod
    def build_path(path):
        """
        Generates an absolute path if a relative is passed.
        """
        if not os.path.isabs(path):
            path = os.path.join(Configuration.working_directory, path)
        return path
