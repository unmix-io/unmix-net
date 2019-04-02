#!/usr/bin/env python3
# coding: utf8

"""
Helps transposing axis of a multidimensional matrix.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"


TRANSPOSE_DIMENSIONS = [[0], [1, 0], [1, 0, 2], [2, 0, 1, 3]]


def transpose_step(input):
   return input.transpose(*TRANSPOSE_DIMENSIONS[len(input.shape) - 1])


def pre_post_transpose(input, func):
   return transpose_step(func(transpose_step(input)))