#!/usr/bin/env python3
# coding: utf8

"""
Logs prettified information to console and output file.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"


import os
import colorama
import time as sys_time

from unmix.source.configuration import Configuration
from unmix.source.helpers import converter
from unmix.source.logging.colors import Colors


class Logger(object):

    timers = {}
    log_file = ''

    @staticmethod
    def initialize(write_file=True):
        Logger.timers = {}
        if write_file:
            Logger.log_file = os.path.join(
                Configuration.output_directory, Configuration.get('environment.log_file'))
        colorama.init()

    @staticmethod
    def format(iterable):
        return " ".join(str(i) for i in iterable)

    @staticmethod
    def h1(*args):
        print(Colors.DK_WHITE + Logger.format(args) + Colors.END)
        Logger._write_log("", Logger.format(args))

    @staticmethod
    def wait(*args):
        input(Colors.CYAN + Logger.format(args) + Colors.END)
        Logger._write_log("Waiting", Logger.format(args))

    @staticmethod
    def info(*args):
        print(Colors.DIM + "\t", Logger.format(args), Colors.END)
        Logger._write_log("Info", Logger.format(args))

    @staticmethod
    def debug(*args):
        print(Colors.GREEN + "\t", Logger.format(args), Colors.END)
        Logger._write_log("Debug", Logger.format(args))

    @staticmethod
    def warn(*args):
        print(Colors.DK_MAGENTA + "WARN:\t" + Colors.END +
              Colors.MAGENTA, Logger.format(args), Colors.END)
        Logger._write_log("Warning", Logger.format(args))

    @staticmethod
    def error(*args):
        print(Colors.DK_RED + Colors.BLINK + "ERROR:\t" +
              Colors.END + Colors.RED, Logger.format(args), Colors.END)
        Logger._write_log("Error", Logger.format(args))

    @staticmethod
    def time(key):
        Logger.timers[key] = sys_time.time()

    @staticmethod
    def time_end(key):
        if key in Logger.timers:
            t = sys_time.time() - Logger.timers[key]
            print("\t" + str(t) + Colors.DIM + " s \t" + key + Colors.END)
            del Logger.timers[key]

    @staticmethod
    def notify(*args):
        # Play bell
        print('\a')

    @staticmethod
    def _write_log(title, message):
        try:
            if not Logger.log_file:
                return
            if title:
                content = "%s\t%s: %s" % (
                    converter.get_timestamp(), title, message)
            else:
                content = "%s\t%s" % (
                    converter.get_timestamp(), message)
            with open(Logger.log_file, "a", newline='\n', encoding='utf8') as file:
                file.write(content + '\n')
        except:
            pass
