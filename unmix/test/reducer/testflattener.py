#!/usr/bin/env python
# coding: utf8

"""
Tests of the flattener mechanism.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"

import numpy as np

from unmix.source.helpers.reducer import lflatter
from unmix.source.helpers.reducer import rflatter


def test_lflatter_3_1():
    input = np.empty((1,129,769))
    output = lflatter(input, 1)
    assert len(output.shape) == len(input.shape) - 1
    assert input.shape[0] * input.shape[1] == output.shape[0]


def test_lflatter_5_1():
    input = np.empty((1,129,769,64,2))
    output = lflatter(input, 1)
    assert len(output.shape) == len(input.shape) - 1
    assert input.shape[0] * input.shape[1] == output.shape[0]


def test_lflatter_3_2():
    input = np.empty((1,129,769))
    output = lflatter(input, 2)
    assert len(output.shape) == len(input.shape) - 2
    assert input.shape[0] * input.shape[1] * input.shape[2] == output.shape[0]


def test_lflatter_5_2():
    input = np.empty((1,129,769,64,2))
    output = lflatter(input, 2)
    assert len(output.shape) == len(input.shape) - 2
    assert input.shape[0] * input.shape[1] * input.shape[2] == output.shape[0]


def test_rflatter_3_1():
    input = np.empty((1,129,769))
    output = rflatter(input, 1)
    assert len(output.shape) == len(input.shape) - 1
    assert input.shape[-1] * input.shape[-2] == output.shape[-1]


def test_rflatter_5_1():
    input = np.empty((1,129,769,64,2))
    output = rflatter(input, 1)
    assert len(output.shape) == len(input.shape) - 1
    assert input.shape[-1] * input.shape[-2] == output.shape[-1]


def test_rflatter_3_2():
    input = np.empty((1,129,769))
    output = rflatter(input, 2)
    assert len(output.shape) == len(input.shape) - 2
    assert input.shape[-1] * input.shape[-2] * input.shape[-3] == output.shape[-1]


def test_rflatter_5_2():
    input = np.empty((1,129,769,64,2))
    output = rflatter(input, 2)
    assert len(output.shape) == len(input.shape) - 2
    assert input.shape[-1] * input.shape[-2] * input.shape[-3] == output.shape[-1]


if __name__ == "__main__":
    test_lflatter_3_1()
    test_lflatter_5_1()
    test_lflatter_3_2()
    test_lflatter_5_2()
    test_rflatter_3_1()
    test_rflatter_5_1()
    test_rflatter_5_2()
