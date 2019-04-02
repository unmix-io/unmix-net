#!/usr/bin/env python3
# coding: utf8

"""
Pipeline dimension for handling input or target data.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"


class PipelineDimension:

    def __init__(self, name, chop_size, transform):
        self.name = name
        self.chop_size = chop_size
        self.transformer = getattr(self, transform)

    def magnitude_normalize(self, input):
        return None

    def magnitude_mask(self, input):
        return None
