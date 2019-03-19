"""
Loads and handels training and validation data collections.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"


from unmix.source.configuration import Configuration
from unmix.source.exceptions.configurationerror import ConfigurationError


class DataCollectionHandler(object):

    @staticmethod
    def load_training():
        return [], []

    @staticmethod
    def load_validation():
        return [], []