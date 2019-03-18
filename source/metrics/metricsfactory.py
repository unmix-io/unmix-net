
"""
Builds metrics from configuration.
"""


import keras.backend as keras

from configuration import Configuration
from exceptions.configurationerror import ConfigurationError


class MetricsFactory(object):

    @staticmethod
    def build():
        configs = Configuration.get('training.metrics', False)
        metrics = []
        for config in configs:
            metrics.append(getattr(MetricsFactory, config))
        return metrics

    @staticmethod
    def mean_pred(y_true, y_pred):
        return keras.mean(y_pred)

    @staticmethod
    def max_pred(y_true, y_pred):
        return keras.max(y_pred)
