#!/usr/bin/env python3
# coding: utf8

"""
Builds choppers to chop the input spectrograms.
"""

from unmix.source.choppers.chopper import Chopper
from unmix.source.configuration import Configuration


class CutterFactory(object):

    @staticmethod
    def build():
        config = Configuration.get('training.cutter')
        if config:
            return Cutter(config.position, config.size,)
        return None
