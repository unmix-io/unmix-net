#!/usr/bin/env python3
# coding: utf8

"""
Builds a pipline for data preprocessing.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"


import keras.backend as keras

from unmix.source.configuration import Configuration
from unmix.source.pipline.pipeline import Pipeline
from unmix.source.exceptions.configurationerror import ConfigurationError


class PipelineFactory(object):

    @staticmethod
    def build():
        input_config = Configuration.get('pipeline.input', False)

        input_pipeline = Pipeline("input", input_config.chop, input_config.transform)
        target_config = Configuration.get('pipeline.input', False)
        target_pipeline = Pipeline("target", target_config.chop, target_config.transform)
        
        return input_pipeline, target_pipeline

    