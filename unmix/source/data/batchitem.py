#!/usr/bin/env python3
# coding: utf8

"""
Batch item with song and chop offset.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"


class BatchItem(object):

    def __init__(self, song, index):
        self.song = song
        self.index = index
        self.name = '%s-%i' % (song.name, index)

    def load(self):
        return self.song.load()
