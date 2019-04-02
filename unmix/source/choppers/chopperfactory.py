#!/usr/bin/env python3
# coding: utf8

"""
Builds a chopper to chop the input data.
"""

from unmix.source.choppers.chopper import Chopper
from unmix.source.choppers.emptychopper import EmptyChopper
from unmix.source.configuration import Configuration


class ChopperFactory(object):

    @staticmethod
    def build():
        config = Configuration.get('training.chopper')
        if config:
            return Chopper(config.direction, config.mode, config.size)
        return EmptyChopper()

