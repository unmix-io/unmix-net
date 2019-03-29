#!/usr/bin/env python
# coding: utf8

"""
Builds choppers to chop the input spectrograms.
"""

from unmix.source.choppers.chopper import Chopper
from unmix.source.configuration import Configuration


class ChoppersFactory(object):

    @staticmethod
    def build():
        config = Configuration.get('training.choppers')
        choppers = []
        if hasattr(config,'horizontal'):
            choppers.append(Chopper(Chopper.DIRECTION_HORIZONTAL, config.horizontal.mode, config.horizontal.size))
        if hasattr(config,'vertical'):
            choppers.append(Chopper(Chopper.DIRECTION_VERTICAL, config.vertical.mode, config.vertical.size))
        return choppers
