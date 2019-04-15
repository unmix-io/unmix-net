#!/usr/bin/env python3
# coding: utf8

"""
Pretty colors for console output.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"


class Colors:
    END = "\033[0m"
    BRIGHT = "\033[1m"
    DIM = "\033[2m"
    UNDERSCORE = "\033[4m"
    BLINK = "\033[5m"

    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    DK_RED = "\033[41m"
    DK_GREEN = "\033[42m"
    DK_YELLOW = "\033[43m"
    DK_BLUE = "\033[44m"
    DK_MAGENTA = "\033[45m"
    DK_CYAN = "\033[46m"
    DK_WHITE = "\033[47m"
    
    CLEAR_SCREEN = '\033[2J'