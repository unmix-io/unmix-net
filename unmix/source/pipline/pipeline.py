#!/usr/bin/env python3
# coding: utf8

"""
Pipeline for preprocessing data.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"


from unmix.source.pipline.choppers.chopper import Chopper


class Pipeline:

    def __init__(self, chop_step, save_audio):
        self.chopper = Chopper(chop_step)
        self.save_audio = save_audio
        self.input_dimension = None
        self.target_dimension = None

    def apply(self, mix, vocals, index):
        x, y, xy = self.get_chops(mix, vocals, index)
        return self.transform(x, y)

    def get_chops(self, mix, vocals, index):
        return self.chopper.get_chop(mix, index, self.input_dimension.chop_size), \
            self.chopper.get_chop(vocals, index, self.target_dimension.chop_size), \
            self.chopper.get_chop(
                mix, index, self.target_dimension.chop_size) if not self.dimensions_match() else None

    def transform(self, x, y):
        return self.input_dimension.transformer.transform(x, y), self.target_dimension.transformer.transform(x, y)

    def dimensions_match(self):
        return self.input_dimension.chop_size == self.target_dimension.chop_size
