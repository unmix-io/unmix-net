"""
Converts and handles data and types.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"


import os
import datetime

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
    return value and value.lower() in ("yes", "true", "t", "1", "y")


def get_timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")