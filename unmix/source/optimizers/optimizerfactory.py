#!/usr/bin/env python3
# coding: utf8

"""
Builds an optimizer from configuration.
"""


from keras.optimizers import Adam, RMSprop

from unmix.source.configuration import Configuration


class OptimizerFactory(object):

    @staticmethod
    def build():
        optimizer = Configuration.get('training.optimizer.name')
        return getattr(OptimizerFactory, optimizer)(OptimizerFactory, Configuration.get('training.optimizer'))

    def adam(self, options):
        return Adam(lr=options.lr)

    def rmsprop(self, options):
        return RMSprop(options.lr)
