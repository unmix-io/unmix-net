"""
Tests of the flattener mechanism.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"

import numpy as np

from unmix.source.helpers.reducer import lflatter


def test_lflatter_1():
    matrix = np.array([[["000","001"],["010","011"],["020","021"],["030","031"],["040","051"],["060","071"]],
                       [["100","101"],["110","111"],["120","121"],["130","131"],["140","151"],["160","171"]],
                       [["200","201"],["210","211"],["220","221"],["230","231"],["240","251"],["260","271"]],
                       [["300","301"],["310","311"],["320","321"],["330","331"],["340","351"],["360","371"]]])
    output = lflatter(matrix, 1)
    assert len(output.shape) == 2


def test_lflatter_2():
    matrix = np.array([[["000","001"],["010","011"],["020","021"],["030","031"],["040","051"],["060","071"]],
                       [["100","101"],["110","111"],["120","121"],["130","131"],["140","151"],["160","171"]],
                       [["200","201"],["210","211"],["220","221"],["230","231"],["240","251"],["260","271"]],
                       [["300","301"],["310","311"],["320","321"],["330","331"],["340","351"],["360","371"]]])
    output = lflatter(matrix, 2)
    assert len(output.shape) == 1


if __name__ == "__main__":
    test_lflatter_1()
    test_lflatter_2()
