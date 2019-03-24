"""
Helps reducing and handle objects.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"


import numpy as np
import functools


def rgetattr(obj, attr, *args):

    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))

def lflatter(input, times):
    if len(input.shape) == 2:
        output = input.flatten()
    else:
        output = input.reshape(-1, *input.shape[-2:])
    if times > 1:
        return lflatter(output, times - 1)
    return output
