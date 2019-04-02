#!/usr/bin/env python3
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

    def load(self, choppers=[]):
        return self.song.load(choppers, self.offset)
