"""
Tests of the chopper objects.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"


import numpy as np

from unmix.source.choppers.chopper import Chopper


def test_vertical_split_100_10():
    chopper = Chopper(Chopper.DIRECTION_VERTICAL, Chopper.MODE_SPLIT, 10)
    chops = chopper.chop(range(100))
    assert len(chops) == 10


def test_vertical_split_105_10():
    chopper = Chopper(Chopper.DIRECTION_VERTICAL, Chopper.MODE_SPLIT, 10)
    chops = chopper.chop(range(105))
    assert len(chops) == 10


def test_vertical_overlap_100_10():
    chopper = Chopper(Chopper.DIRECTION_VERTICAL, Chopper.MODE_OVERLAP, 10)
    chops = chopper.chop(range(100))
    assert len(chops) == 19


def test_vertical_split_matrix3():
    matrix = np.array([[["000","001"],["010","011"],["020","021"],["030","031"],["040","051"],["060","071"]],
                       [["100","101"],["110","111"],["120","121"],["130","131"],["140","151"],["160","171"]],
                       [["200","201"],["210","211"],["220","221"],["230","231"],["240","251"],["260","271"]],
                       [["300","301"],["310","311"],["320","321"],["330","331"],["340","351"],["360","371"]]])
    chopper = Chopper(Chopper.DIRECTION_VERTICAL, Chopper.MODE_SPLIT, 2)
    chops = chopper.chop(matrix)
    assert len(chops) == 2


def test_horizontal_split_matrix3():
    matrix = np.array([[["000","001"],["010","011"],["020","021"],["030","031"],["040","051"],["060","071"]],
                       [["100","101"],["110","111"],["120","121"],["130","131"],["140","151"],["160","171"]],
                       [["200","201"],["210","211"],["220","221"],["230","231"],["240","251"],["260","271"]],
                       [["300","301"],["310","311"],["320","321"],["330","331"],["340","351"],["360","371"]]])
    chopper = Chopper(Chopper.DIRECTION_HORIZONTAL, Chopper.MODE_SPLIT, 2)
    chops = chopper.chop(matrix)
    assert len(chops) == 3


def test_vertical_horizontal_split_matrix3():
    matrix = np.array([[["000","001"],["010","011"],["020","021"],["030","031"],["040","051"],["060","071"]],
                       [["100","101"],["110","111"],["120","121"],["130","131"],["140","151"],["160","171"]],
                       [["200","201"],["210","211"],["220","221"],["230","231"],["240","251"],["260","271"]],
                       [["300","301"],["310","311"],["320","321"],["330","331"],["340","351"],["360","371"]]])
    vertical_chopper = Chopper(Chopper.DIRECTION_VERTICAL, Chopper.MODE_SPLIT, 2)
    chops = vertical_chopper.chop(matrix)
    horizontal_chopper = Chopper(Chopper.DIRECTION_HORIZONTAL, Chopper.MODE_SPLIT, 2)
    chops = horizontal_chopper.chop(chops)
    assert len(chops) == 3


if __name__ == "__main__":
    #test_horizontal_split_100_10()
    #test_horizontal_split_105_10()
    #test_horizontal_overlap_100_10()
    test_vertical_split_matrix3()
    test_horizontal_split_matrix3()
    test_vertical_horizontal_split_matrix3()
