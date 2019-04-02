#!/usr/bin/env python3
# coding: utf8

"""
Tests of the cutter objects.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"


import numpy as np

from unmix.source.cutters.cutter import Cutter


def test_left_1_1d():
    input = np.array(range(10))
    cutter = Cutter(Cutter.POSITION_LEFT, 1)
    cut = cutter.cut(input)
    assert len(cut) == 1
    assert cut[0] == input[0]


def test_left_1_2d():
    input = np.array([range(10)])
    cutter = Cutter(Cutter.POSITION_LEFT, 1)
    cut = cutter.cut(input)
    assert cut.shape[1] == 1
    assert cut[0][0] == input[0][0]


def test_right_1_1d():
    input = np.array(range(10))
    cutter = Cutter(Cutter.POSITION_RIGHT, 1)
    cut = cutter.cut(input)
    assert len(cut) == 1
    assert cut[0] == input[-1]


def test_right_1_2d():
    input = np.array([range(10)])
    cutter = Cutter(Cutter.POSITION_RIGHT, 1)
    cut = cutter.cut(input)
    assert cut.shape[1] == 1
    assert cut[0][0] == input[0][-1]


def test_center_1_1d():
    input = np.array(range(10))
    cutter = Cutter(Cutter.POSITION_CENTER, 1)
    cut = cutter.cut(input)
    assert len(cut) == 1
    assert cut[0] == input[4]


def test_center_1_2d():
    input = np.array([range(10)])
    cutter = Cutter(Cutter.POSITION_CENTER, 1)
    cut = cutter.cut(input)
    assert cut.shape[1] == 1
    assert cut[0][0] == input[0][4]


if __name__ == "__main__":
    test_left_1_1d()
    test_left_1_2d()
    test_right_1_2d()
    test_right_1_2d()
    test_center_1_1d()
    test_center_1_2d()
    print("Test run successful.")
