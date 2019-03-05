
"""
Builds a keras loss function from configuration.
"""


from keras import losses

from configuration import Configuration
from exceptions.configurationerror import ConfigurationError

class LossFunctionFactory(object):

    @staticmethod
    def build():
        loss_function = Configuration.get('training.loss_function', False)
        return getattr(LossFunctionFactory, loss_function)

    def mean_squared_error(self, y_true, y_pred):
        return losses.mean_squared_error(y_true, y_pred)

    def mean_absolute_error(self, y_true, y_pred):
        return losses.mean_absolute_error(y_true, y_pred)

    def mean_squared_log_error(self, y_true, y_pred):
        return losses.mean_squared_logarithmic_error(y_true, y_pred)
