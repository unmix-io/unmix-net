
"""
Builds choppers to chop the input spectrograms.
"""

from unmix.source.configuration import Configuration
from unmix.source.exceptions.configurationerror import ConfigurationError


class Chopper:

    DIRECTION_HORIZONTAL = "horizontal"
    DIRECTION_VERTICAL = "vertical"
    MODE_SPLIT = "split"
    MODE_OVERLAP = "overlap"

    def __init__(self, direction, mode, size):
        self.direction = direction
        self.mode = mode
        self.size = size

    def chop(self, input):
        chops = []
        if self.direction == Chopper.DIRECTION_HORIZONTAL:
            if self.mode == Chopper.MODE_SPLIT:
                slices = int(len(input) / self.size)
                for i in range(slices):
                    chops.append(input[i * self.size : (i+1) * self.size])
        return chops