#!/usr/bin/env python3
# coding: utf8

"""
Builds metrics from configuration.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"


import keras.backend as keras

from unmix.source.configuration import Configuration


class Accuracy(object):

    def __init__(self, engine):
        self.engine = engine

    def evaluate(self):
        return None
