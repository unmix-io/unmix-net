#!/usr/bin/env python3
# coding: utf8

"""
Helps transposing axis of a multidimensional matrix.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"


PRE_TRANSPOSE_DIMENSIONS = [[0], [1, 0], [1, 0, 2], [2, 0, 1, 3]]
POST_TRANSPOSE_DIMENSIONS = [[0], [0, 1], [0, 2, 1], [0, 2, 1, 3], [0, 2, 1, 3, 4]]


def pre_transpose(input):
   return input.transpose(*PRE_TRANSPOSE_DIMENSIONS[len(input.shape) - 1])

def post_transpose(input):
   return input.transpose(*POST_TRANSPOSE_DIMENSIONS[len(input.shape) - 1])

def pre_post_transpose(input, func):
   return post_transpose(func(pre_transpose(input)))