#!/usr/bin/env python3
# coding: utf8

"""
Tests the memory usage monitor.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"


import numpy as np

from unmix.source.helpers.memorymonitor import track


@track
def test_list_create():
    x = [1] * 1000
    return x


if __name__ == "__main__":
    test_list_create()
    print("Test run successful.")
