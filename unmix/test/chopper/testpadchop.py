#!/usr/bin/env python3
# coding: utf8

"""
Tests of the chopper padding.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"


import numpy as np

from unmix.source.pipeline.choppers.chopper import Chopper
import unmix.source.helpers.transposer as transposer


def test_pad_none():
    input = np.array([range(1,10)])
    chopper = Chopper(10)
    chop = transposer.pre_post_transpose(input, lambda x: chopper.pad_chop(x, 0, 2))
    assert len(chop) == 2
    assert chop[0] != 0
    assert chop[-1] != 0


def test_pad_left():
    input = np.array([range(1,10)])
    chopper = Chopper(10)
    chop = transposer.pre_post_transpose(input, lambda x: chopper.pad_chop(x, -2, 2))
    assert len(chop) == 4
    assert chop[0] == 0
    assert chop[1] == 0
    assert chop[2] != 0
    assert chop[3] != 0


def test_pad_right():
    input = np.array([range(1,10)])
    chopper = Chopper(10)
    chop = transposer.pre_post_transpose(input, lambda x: chopper.pad_chop(x, 7, 11))
    assert len(chop) == 4
    assert chop[0] != 0
    assert chop[1] != 0
    assert chop[2] == 0
    assert chop[3] == 0


def test_pad_left_right():
    input = np.array([range(1,3)])
    chopper = Chopper(10)
    chop = transposer.pre_post_transpose(input, lambda x: chopper.pad_chop(x, -2, 4))
    assert len(chop) == 6
    assert chop[0] == 0
    assert chop[1] == 0
    assert chop[2] != 0
    assert chop[3] != 0
    assert chop[4] == 0
    assert chop[5] == 0


if __name__ == "__main__":
    test_pad_none()
    test_pad_left()
    test_pad_right()
    test_pad_left_right()
    print("Test run successful.")
