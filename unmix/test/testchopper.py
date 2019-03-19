"""
Tests of the chopper objects.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"


from unmix.source.choppers.chopper import Chopper
from unmix.source.configuration import Configuration


def test_horizontal_split_100_10():
    chopper = Chopper(Chopper.DIRECTION_HORIZONTAL, Chopper.MODE_SPLIT, 10)
    chops = chopper.chop(range(100))
    assert len(chops) == 10


def test_horizontal_split_105_10():
    chopper = Chopper(Chopper.DIRECTION_HORIZONTAL, Chopper.MODE_SPLIT, 10)
    chops = chopper.chop(range(105))
    assert len(chops) == 10


def test_horizontal_overlap_100_10():
    chopper = Chopper(Chopper.DIRECTION_HORIZONTAL, Chopper.MODE_OVERLAP, 10)
    chops = chopper.chop(range(100))
    assert len(chops) == 19


if __name__ == "__main__":
    test_horizontal_split_100_10()
    test_horizontal_split_105_10()
    test_horizontal_overlap_100_10()
