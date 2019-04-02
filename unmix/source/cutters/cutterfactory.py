#!/usr/bin/env python3
# coding: utf8

"""
Builds a cutter to cut from the input data.
"""

from unmix.source.cutters.cutter import Cutter
from unmix.source.configuration import Configuration


class CutterFactory(object):

    @staticmethod
    def build():
        config = Configuration.get('training.cutter')
        if config:
            return Cutter(config.position, config.size,)
        return None
