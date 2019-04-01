#!/usr/bin/env python3
# coding: utf8

"""
Model of the unmix.io neuronal network learning object.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"


import os
import keras.utils

from unmix.source.callbacks.callbacksfactory import CallbacksFactory
from unmix.source.choppers.choppersfactory import ChoppersFactory
from unmix.source.configuration import Configuration
from unmix.source.data.datagenerator import DataGenerator
from unmix.source.data.dataloader import DataLoader
from unmix.source.helpers import console
from unmix.source.helpers import converter
from unmix.source.lossfunctions.lossfunctionfactory import LossFunctionFactory
from unmix.source.metrics.metricsfactory import MetricsFactory
from unmix.source.models.modelfactory import ModelFactory
from unmix.source.optimizers.optimizerfactory import OptimizerFactory
from unmix.source.normalizers.normalizerfactory import NormalizerFactory


class Engine:

    def __init__(self):
        optimizer = OptimizerFactory.build()
        loss_function = LossFunctionFactory.build()
        metrics = MetricsFactory.build()
        normalizer = NormalizerFactory.build()
        self.callbacks = CallbacksFactory.build()
        choppers = ChoppersFactory.build()

        training_songs, validation_songs = DataLoader.load()

        self.model = ModelFactory.build()
        self.model.compile(loss=loss_function,
                           optimizer=optimizer, metrics=metrics)
        self.model.summary(Configuration.get(
            'environment.summary_line_length'))
        self.plot_model()
        console.debug("Model initialized with %d parameters." %
                      self.model.count_params())

        self.training_generator = DataGenerator(training_songs, choppers, normalizer)
        self.validation_generator = DataGenerator(validation_songs, choppers, normalizer)

    def plot_model(self):
        try:
            path = Configuration.get_path('environment.model_plot_folder')
            if path:
                name = Configuration.get('training.model').name
                file_name = os.path.join(
                    path, ('%s_%s-model.png' % (converter.get_timestamp(), name)))
                keras.utils.plot_model(self.model, file_name)
        except Exception as e:
            console.error("Error while plotting model: %s." % str(e))

    def train(self, epoch_start=0):
        epoch_count = Configuration.get('training.epoch.count')
        history = self.model.fit_generator(
            generator=self.training_generator,
            validation_data=self.validation_generator,
            initial_epoch=epoch_start,
            epochs=epoch_start + epoch_count,
            shuffle=False,
            verbose=Configuration.get('training.verbose'),
            callbacks=self.callbacks)
        self.save_weights()
        return history

    def predict(self, data):
        y = self.model.predict(data)
        return y

    def save_weights(self):
        path = os.path.join(Configuration.get_path(
            'environment.weights.folder'), Configuration.get('environment.weights.file'))
        console.info("Saved weights to: %s" % path)
        self.model.save_weights(path, overwrite=True)

    def load_weights(self, path=None):
        if not path:            
            path = os.path.join(Configuration.get_path('environment.weights.folder'), Configuration.get('environment.weights.file'))
        console.info("Load weights from: %s" % path)
        self.model.load_weights(path)
