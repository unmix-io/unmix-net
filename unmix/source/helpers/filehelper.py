#!/usr/bin/env python3
# coding: utf8

"""
Helps handling files and folders.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"


import glob
import numpy as np
import os


def build_abspath(path, working_directory=''):
    if not os.path.isabs(path):
        if not working_directory:
            working_directory = os.getcwd()
        path = os.path.join(working_directory, path)
    return path


def get_latest(folder, pattern):
    """
    Gets the latest file in a folder matching the naming pattern.
    """
    files = glob.glob(os.path.join(folder, pattern))
    return max(files, key=os.path.getctime)