
"""
Builds choppers to chop the input spectrograms.
"""

from choppers.chopper import Chopper
from configuration import Configuration


class ChoppersFactory(object):

    @staticmethod
    def build():
        configs = Configuration.get('training.loss_function')
        choppers = []
        for config in configs:
            choppers.append(Chopper(config.direction, config.mode, config.size))
        return choppers
