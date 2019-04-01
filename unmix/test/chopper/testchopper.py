#!/usr/bin/env python3
# coding: utf8

"""
Tests of the chopper objects.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"


import numpy as np

from unmix.source.choppers.chopper import Chopper


def test_horizontal_overlap_2d_2():
    input = np.array([range(9)])
    chopper = Chopper(Chopper.DIRECTION_HORIZONTAL, Chopper.MODE_OVERLAP, 2)
    chops = chopper.chop(input)
    assert len(chops) == 8
    input = np.array([range(10)])
    chopper = Chopper(Chopper.DIRECTION_HORIZONTAL, Chopper.MODE_OVERLAP, 2)
    chops = chopper.chop(input)
    assert len(chops) == 9
    input = np.array([range(11)])
    chopper = Chopper(Chopper.DIRECTION_HORIZONTAL, Chopper.MODE_OVERLAP, 2)
    chops = chopper.chop(input)
    assert len(chops) == 10


def test_horizontal_overlap_2d_4():
    input = np.array([range(11)])
    chopper = Chopper(Chopper.DIRECTION_HORIZONTAL, Chopper.MODE_OVERLAP, 4)
    chops = chopper.chop(input)
    assert len(chops) == 4
    input = np.array([range(12)])
    chopper = Chopper(Chopper.DIRECTION_HORIZONTAL, Chopper.MODE_OVERLAP, 4)
    chops = chopper.chop(input)
    assert len(chops) == 5
    input = np.array([range(13)])
    chopper = Chopper(Chopper.DIRECTION_HORIZONTAL, Chopper.MODE_OVERLAP, 4)
    chops = chopper.chop(input)
    assert len(chops) == 5


def test_horizontal_stepwise_2d_2():
    input = np.array([range(9)])
    chopper = Chopper(Chopper.DIRECTION_HORIZONTAL, Chopper.MODE_STEPWISE, 3)
    chops = chopper.chop(input)
    assert len(chops) == 7
    input = np.array([range(10)])
    chopper = Chopper(Chopper.DIRECTION_HORIZONTAL, Chopper.MODE_STEPWISE, 3)
    chops = chopper.chop(input)
    assert len(chops) == 8


def test_horizontal_split_3d_64():
    input = np.empty((769, 2000, 2))
    chopper = Chopper(Chopper.DIRECTION_HORIZONTAL, Chopper.MODE_SPLIT, 64)
    chops = chopper.chop(input)
    assert len(chops) == 31  # floor(2000/64)


def test_horizontal_overlap_3d_64():
    input = np.empty((769, 2000, 2))
    chopper = Chopper(Chopper.DIRECTION_HORIZONTAL, Chopper.MODE_OVERLAP, 64)
    chops = chopper.chop(input)
    assert len(chops) == 61  # floor((2000-32)/(32))


def test_horizontal_stepwise_3d_64():
    input = np.empty((769, 2000, 2))
    chopper = Chopper(Chopper.DIRECTION_HORIZONTAL, Chopper.MODE_STEPWISE, 64)
    chops = chopper.chop(input)
    assert len(chops) == 1937  # 2000 - 64 + 1


if __name__ == "__main__":
    test_horizontal_overlap_2d_2()
    test_horizontal_overlap_2d_4()
    test_horizontal_stepwise_2d_2()
    test_horizontal_split_3d_64()
    test_horizontal_overlap_3d_64()
    test_horizontal_stepwise_3d_64()
    print("Test run successful.")
