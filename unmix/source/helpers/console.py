#!/usr/bin/env python3
# coding: utf8

"""
Prints pretty console outputs.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"


import colorama
import time as sys_time

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
timers = {}
colorama.init()


def fmt(iterable):
    return " ".join(str(i) for i in iterable)


def h1(*args):
    print(Colors.BRIGHT, fmt(args), Colors.END)


def wait(*args):
    input(Colors.BLUE + fmt(args) + Colors.END)


def log(*args):
    print(Colors.YELLOW, fmt(args), Colors.END)


def info(*args):
    print(Colors.DIM + "\t", fmt(args), Colors.END)


def debug(*args):
    print(Colors.DK_BLUE + "\t", fmt(args), Colors.END)


def warn(*args):
    print(Colors.DK_CYAN + "WARN:\t" + Colors.END +
          Colors.CYAN, fmt(args), Colors.END)


def error(*args):
    print(Colors.DK_RED + Colors.BLINK + "ERROR:\t" +
          Colors.END + Colors.RED, fmt(args), Colors.END)


def time(key):
    timers[key] = sys_time.time()


def time_end(key):
    if key in timers:
        t = sys_time.time() - timers[key]
        print("\t" + str(t) + Colors.DIM + " s \t" + key + Colors.END)
        del timers[key]


def notify(*args):
    # Play bell
    print('\a')
