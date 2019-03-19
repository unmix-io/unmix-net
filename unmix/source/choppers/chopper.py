
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
        if self.direction == Chopper.DIRECTION_HORIZONTAL:
            if self.mode == Chopper.MODE_SPLIT:
                return self.chop_horizontal_split(input)
            if self.mode == Chopper.MODE_OVERLAP:
                return self.chop_horizontal_overlap(input)
        raise ConfigurationError("Chopper with invalid configuration")

    def chop_horizontal_split(self, input):
        chops = []
        slices = int(len(input) / self.size)
        for i in range(slices):
            chops.append(input[(i * self.size):((i+1) * self.size)])
        return chops

    def chop_horizontal_overlap(self, input):
        chops = []
        position = 0
        step = int(self.size / 2)
        while position + step < len(input):
            chops.append(input[position:(position + self.size)])
            position += step
        return chops
