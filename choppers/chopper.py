
"""
Builds choppers to chop the input spectrograms.
"""

from configuration import Configuration
from exceptions.configurationerror import ConfigurationError

class Chopper:

    def __init__(self, direction, mode, size):
        self.direction = direction
        self.mode = mode
        self.size = size

    def chop(input):
        return input # TODO Chop horizontal or vertically