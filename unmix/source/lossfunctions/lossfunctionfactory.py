
"""
Builds a keras loss function from configuration.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"


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
