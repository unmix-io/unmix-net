#!/usr/bin/env python3
# coding: utf8

"""
Tests of the chopper objects.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"


import numpy as np

from unmix.source.pipeline.choppers.chopper import Chopper

def test_start():
    input = np.ones((1000,64))
    chopper = Chopper(64)
    chop = chopper.chop_n_pad(input, 0, 64)
    # When we access the first item, the chopper adds padding of -size - should be zero
    assert chop.shape[1] == 64
    assert chop.shape[0] == 1000
    assert np.all(chop[:, 0:32] == 0)
    assert np.all(chop[:, 32:] == 1)


def test_end():
    input = np.ones((1000,10))
    chopper = Chopper(10)
    chop = chopper.chop_n_pad(input, 1, 10)
    assert chop.shape[1] == 10
    assert chop.shape[0] == 1000
    assert np.all(chop[:, 0:5] == 1)
    assert np.all(chop[:, 5:] == 0)

def test_end_odd():
    input = np.ones((1000,9))
    chopper = Chopper(7)
    chop = chopper.chop_n_pad(input, 1, 5)
    assert chop.shape[1] == 5
    assert chop.shape[0] == 1000
    assert np.all(chop[:, 0:4] == 1)
    assert np.all(chop[:, 4:] == 0)

def test_middle():
    input = np.ones((1000,2000))
    chopper = Chopper(10)
    chop = chopper.chop_n_pad(input, 35, 10)
    assert chop.shape[1] == 10
    assert chop.shape[0] == 1000
    assert np.all(chop[:, :] == 1)


def test_lengths():
    chopper = Chopper(2)
    assert chopper.calculate_chops(4, 2) == 3
    assert chopper.calculate_chops(5, 2) == 3
    assert chopper.calculate_chops(6, 2) == 4
    assert chopper.calculate_chops(7, 2) == 4
    assert chopper.calculate_chops(8, 2) == 5

def test_chopping():
    size = 10
    input = np.ones((1000,1998))
    chopper = Chopper(size)
    chop_count = chopper.calculate_chops(input.shape[1], size)

    for i in range(chop_count):
        chop = chopper.chop_n_pad(input, i, size)
        assert(chop.shape[1] == size)
        if i == 0:
            assert np.all(chop[:, 0:int(size/2)] == 0)
            assert np.all(chop[:, int(size/2):] == 1)
        elif i == chop_count - 1:
            remainder = (input.shape[1] - int(size/2)) % size
            assert np.all(chop[:, 0:remainder] == 1)
            assert np.all(chop[:, remainder:] == 0)
        else:
            assert np.all(chop[:, :] == 1)

if __name__ == "__main__":
    test_start()
    test_end()
    test_end_odd()
    test_middle()
    test_lengths()
    test_chopping()
    print("Test run successful.")
