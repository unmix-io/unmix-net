#!/usr/bin/env python3
# coding: utf8

"""
Monitors the memory usage of the application.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"


import os
import time

from unmix.source.logging.logger import Logger
from unmix.source.helpers import converter


def get_process_memory():
    import psutil
    process = psutil.Process(os.getpid())
    return process.memory_full_info().rss


def track(func):
    def wrapper(*args, **kwargs):
        memory_before = get_process_memory() / 1000
        start = time.time()
        result = func(*args, **kwargs)
        elapsed_time = converter.elapsed_since(start)
        memory_after = get_process_memory() / 1000

        Logger.debug("Memory usage profile for '%s':\n\t" % func.__name__
                      + "Before: %d\n\t" % memory_before
                      + "After: %d\n\t" % memory_after
                      + "Consumed: %d\n\t" % (memory_after - memory_after)
                      + "Runtime: %s\n\t" % elapsed_time)
        return result
    return wrapper
