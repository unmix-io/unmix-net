"""
Thrown if configuration is missing or invalid.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"


class ConfigurationError(Exception):

    def __init__(self, key, value = False):
        if value:            
            super().__init__("Invalid configuration for key '%s' with value '%s" % (key, value))       
        else:
            super().__init__("Invalid or missing configuration for key '%s'" % key)
