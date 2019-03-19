"""
Model of the unmix.io neuronal network.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"

import os
import keras.utils

from unmix.source.helpers import console
from unmix.source.helpers import reducer
from unmix.source.helpers import converter
from unmix.source.configuration import Configuration
from unmix.source.models.modelfactory import ModelFactory
from unmix.source.metrics.metricsfactory import MetricsFactory
from unmix.source.optimizers.optimizerfactory import OptimizerFactory
from unmix.source.callbacks.callbacksfactory import CallbacksFactory
from unmix.source.data.datacollectionhandler import DataCollectionHandler
from unmix.source.lossfunctions.lossfunctionfactory import LossFunctionFactory


class UnmixNet:

    def __init__(self):
        optimizer = OptimizerFactory.build()
        loss_function = LossFunctionFactory.build()
        metrics = MetricsFactory.build()
        
        self.callbacks = CallbacksFactory.build()

        self.model = ModelFactory.build()
        self.model.compile(loss=loss_function,optimizer=optimizer, metrics=metrics)
        self.model.summary(Configuration.get("environment.summary_line_length"))
        self.plot_model()
        console.debug('Model initialized with %d parameters' %self.model.count_params())

        self.datahandler = DataCollectionHandler()

    def plot_model(self):
        try:
            path = Configuration.get_path("environment.model_plot_folder")
            if path:
                name = Configuration.get('training.model').name
                file_name = os.path.join(path, ("%s-model.png" % name))
                keras.utils.plot_model(self.model, file_name)
        except Exception as e:
            console.error("Error while plotting model: %s" % str(e))

    def train(self, batch_size, epoch_count, epoch_start=0):
        history = self.model.fit_generator(
            generator=
            batch_size=batch_size,
            initial_epoch=epoch_start, epochs=epoch_start + epoch_count,
            validation_data=(x_valid, y_valid),
            callbacks=self.callbacks)

        return False

    def save_weights(self):
        path = Configuration.get_path('environment.weights.file')
        self.model.save_weights(path, overwrite=True)

    def load_weights(self, path):
        path = Configuration.get_path('environment.weights.file')
        self.model.load_weights(path)
