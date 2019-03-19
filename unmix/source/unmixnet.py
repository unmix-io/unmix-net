"""
Model of the unmix.io neuronal network.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"

import os

from unmix.source.helpers import console
from unmix.source.helpers import converter
from unmix.source.configuration import Configuration
from unmix.source.models.modelfactory import ModelFactory
from unmix.source.metrics.metricsfactory import MetricsFactory
from unmix.source.optimizers.optimizerfactory import OptimizerFactory
from unmix.source.lossfunctions.lossfunctionfactory import LossFunctionFactory


class UnmixNet:

    def __init__(self):
        optimizer = OptimizerFactory.build()
        loss_function = LossFunctionFactory.build()
        metrics = MetricsFactory.build()

        self.model = ModelFactory.build()
        self.model.compile(loss=loss_function, optimizer=optimizer, metrics=metrics)
        self.model.summary(Configuration.get("environment.summary_line_length"))
        console.debug('Model initialized with %d parameters' % self.model.count_params())

    def save_weights(self):
        path = Configuration.get_path('environment.weights.file')
        self.model.save_weights(path, overwrite=True)

    def load_weights(self, path):
        path = Configuration.get_path('environment.weights.file')
        self.model.load_weights(path)
