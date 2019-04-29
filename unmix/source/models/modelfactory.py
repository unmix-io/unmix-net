#!/usr/bin/env python3
# coding: utf8

"""
Builds a keras model from configuration.
"""

import importlib

from unmix.source.configuration import Configuration
from unmix.source.exceptions.configurationerror import ConfigurationError
from unmix.source.models.basemodel import BaseModel

# Load all models
from os.path import dirname, basename, isfile
import glob
modules = glob.glob(dirname(__file__)+"/*.py")
for module in [ basename(f)[:-3] for f in modules if isfile(f) and not f.endswith('__init__.py') and not f == "modelfactory"]:
    __import__("unmix.source.models." + module)

class ModelFactory(object):

    @staticmethod
    def build():
        model_config = Configuration.get('training.model', optional=False)
        all_models = BaseModel.__subclasses__()
        if model_config.name:
            return [m().build(model_config) for m in all_models if m.name.lower() == model_config.name.lower()][0]
        raise ConfigurationError('training.model.name')
