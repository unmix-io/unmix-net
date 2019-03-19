"""
Tests of the chopper objects.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"

from unmix.source import *
from unmix.source.choppers.chopper import Chopper
from unmix.source.configuration import Configuration

def test_horizontal_split_100_10():
    chopper = Chopper(Chopper.DIRECTION_HORIZONTAL, Chopper.MODE_SPLIT, 10)
    chopper.chop(range(100))
    


if __name__ == "__main__":
    test_horizontal_split_100_10()
