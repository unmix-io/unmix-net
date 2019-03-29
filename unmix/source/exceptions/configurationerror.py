#!/usr/bin/env python
# coding: utf8

"""
Thrown if configuration is missing or invalid.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"


class ConfigurationError(Exception):

    def __init__(self, key='', value=False):
        if key and value:
            super().__init__("Invalid configuration for key '%s' with value '%s'." % (key, value))
        elif key:
            super().__init__("Invalid or missing configuration for key '%s'." % key)
        else:
            super().__init__("Configuration missing or not initialized.")
