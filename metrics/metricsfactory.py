
"""
Builds an optimizer from configuration.
"""


import keras.backend as keras

from configuration import Configuration
from exceptions.configurationerror import ConfigurationError


class MetricsFactory(object):

    @staticmethod
    def build():
        optimizer = Configuration.get('training.metrics', False)
        return getattr(MetricsFactory, optimizer)

    def mean_pred(self, y_true, y_pred):
        return keras.mean(y_pred)

    def max_pred(self, y_true, y_pred):
        return keras.max(y_pred)
