#!/usr/bin/env python3
# coding: utf8

"""
Batch item with song and chop offset.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"


from unmix.source.choppers.emptychopper import EmptyChopper


class BatchItem(object):

    def __init__(self, song, offset):
        self.song = song
        self.offset = offset

    def load(self, chopper=EmptyChopper()):
        return self.song.load(chopper, self.offset)
