"""
Converts and handles data and types.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"


import os

from configuration import Configuration


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


def build_path(path):
    """
    Generates an absolute path if a relative is passed.
    """
    if not os.path.isabs(path):
        path = os.path.join(Configuration.workingdir, path)
    return path


def str2bool(value):
    """
    Converts a string to a boolean value.
    """
    return value and value.lower() in ("yes", "true", "t", "1", "y")