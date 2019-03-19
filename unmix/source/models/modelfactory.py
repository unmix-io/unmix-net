
"""
Builds a keras model from configuration.
"""


from unmix.source.models import unmix as UnmixModel
from unmix.source.models import leakyrelu as LeakyReLUModel
from unmix.source.models import acapellabot as AcapellaBotModel

from unmix.source.helpers import converter
from unmix.source.configuration import Configuration
from unmix.source.exceptions.configurationerror import ConfigurationError


class ModelFactory(object):

    @staticmethod
    def build():
        model = Configuration.get('training.model', False)
        if model.name:
            if model.name.lower() == LeakyReLUModel.name.lower():
                return LeakyReLUModel.generate(model.alpha1, model.alpha2, model.rate)
            elif model.name.lower() == AcapellaBotModel.name.lower():
                return AcapellaBotModel.generate(0)
            elif model.name.lower() == UnmixModel.name:
                return UnmixModel.generate()
        raise ConfigurationError('training.model.name')
