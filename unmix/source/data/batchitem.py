#!/usr/bin/env python
# coding: utf8

"""
Batch item with song and chop offset.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"


class BatchItem(object):

    def __init__(self, song, offset):
        self.song = song
        self.offset = offset
