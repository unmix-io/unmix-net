#!/usr/bin/env python3
# coding: utf8

"""
Builds metrics from configuration.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"


import keras.backend as keras

from unmix.source.configuration import Configuration


class MetricsFactory(object):

    @staticmethod
    def build():
        configs = Configuration.get('training.metrics', optional=False)
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
