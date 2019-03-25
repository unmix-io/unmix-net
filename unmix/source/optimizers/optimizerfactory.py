
"""
Builds an optimizer from configuration.
"""


from keras.optimizers import Adam, RMSprop

from unmix.source.configuration import Configuration


class OptimizerFactory(object):

    @staticmethod
    def build():
        optimizer = Configuration.get('training.optimizer')
        return getattr(OptimizerFactory, optimizer)(OptimizerFactory)

    def adam(self, **kwargs):
        return Adam(**kwargs)

    def rmsprop(self, **kwargs):
        return RMSprop(**kwargs)
