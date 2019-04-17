#!/usr/bin/env python3
# coding: utf8

"""
Builds a keras loss function from configuration.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"


from keras import backend as K
from keras import losses

from unmix.source.configuration import Configuration


class LossFunctionFactory(object):

    @staticmethod
    def build():
        loss_function = Configuration.get('training.loss_function', False)
        return getattr(LossFunctionFactory, loss_function)

    @staticmethod
    def mean_squared_error(y_true, y_pred):
        return losses.mean_squared_error(y_true, y_pred)

    @staticmethod
    def mean_absolute_error(y_true, y_pred):
        return losses.mean_absolute_error(y_true, y_pred)

    @staticmethod
    def mean_squared_log_error(y_true, y_pred):
        return losses.mean_squared_logarithmic_error(y_true, y_pred)

    @staticmethod
    def squared_hinge(y_true, y_pred):
        return losses.squared_hinge(y_true, y_pred)

    @staticmethod
    def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))

    @staticmethod
    def binary_crossentropy(y_true, y_pred):
        return losses.binary_crossentropy(y_true, y_pred)

    @staticmethod
    def mean_squared_error_noaxis(y_true, y_pred):
        return K.mean(K.square(y_pred - y_true))