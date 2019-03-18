
"""
Builds choppers to chop the input spectrograms.
"""

from choppers.chopper import Chopper
from configuration import Configuration

class ChoppersFactory(object):

    @staticmethod
    def build():
        configurations = Configuration.get('training.loss_function')
        choppers = []
        for config in configurations:
            choppers.append(Chopper(config.direction, config.mode, config.size))
        return choppers