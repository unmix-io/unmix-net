#!/usr/bin/env python3
# coding: utf8

"""
Converts and handles data and types.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"


import datetime
import os
import time
import librosa
import numpy as np

from unmix.source.exceptions.configurationerror import ConfigurationError


def try_eval(expression):
    """
    Returns an evaluated expression if possible.
    If not evaluatable the expression is returned.
    """
    if expression:
        try:
            return eval(expression)
        except:
            pass
    return expression


def str2bool(value):
    """
    Converts a string to a boolean value.
    """
    if type(value) == bool:
        return value
    return value and value.lower() in ('yes', 'true', 't', '1', 'y')


def get_timestamp():
    return datetime.datetime.now().strftime('%Y%m%d-%H%M%S')


def elapsed_since(start, format="%H:%M:%S"):
    return time.strftime(format, time.gmtime(time.time() - start))


def env(key):
    try:
        return os.environ[key]
    except Exception:
        raise ConfigurationError(key)
