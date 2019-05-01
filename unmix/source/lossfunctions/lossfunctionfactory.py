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
    model = False

    @staticmethod
    def build(model):
        LossFunctionFactory.model = model
        loss_function = Configuration.get(
            'training.loss_function', optional=False)
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

    @staticmethod
    def euclidean_loss(x, y):
        return K.sqrt(K.sum(K.square(x - y)))

    @staticmethod
    def L11_loss(y_true, mask_pred):
        return K.sum(K.abs(y_true - LossFunctionFactory.model.input * mask_pred))

    @staticmethod
    def get_members():
        return {
            'mean_squared_error': LossFunctionFactory.mean_squared_error,
            'mean_absolute_error': LossFunctionFactory.mean_absolute_error,
            'mean_squared_log_error': LossFunctionFactory.mean_squared_log_error,
            'squared_hinge': LossFunctionFactory.squared_hinge,
            'root_mean_squared_error': LossFunctionFactory.root_mean_squared_error,
            'binary_crossentropy': LossFunctionFactory.binary_crossentropy,
            'euclidean_loss': LossFunctionFactory.euclidean_loss,
            'L11_loss': LossFunctionFactory.L11_loss
        }
