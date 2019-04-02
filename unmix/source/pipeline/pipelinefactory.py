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
from unmix.source.pipline.pipelinedimension import PipelineDimension
from unmix.source.exceptions.configurationerror import ConfigurationError


class PipelineFactory(object):

    @staticmethod
    def build():
        try:
            chop = Configuration.get('pipeline.chop', False)
            pipeline = Pipeline(chop.step, chop.save_audio)
            input = Configuration.get('pipeline.input', False)
            pipeline.input_dimension = PipelineDimension(
                "input", input.chop.size, input.transform)
            target = Configuration.get('pipeline.target', False)
            pipeline.target_dimension = PipelineDimension(
                "target", target.chop.size. target.transform)
        except Exception as ex:
            raise ConfigurationError('pipeline')
        return pipeline
