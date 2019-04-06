#!/usr/bin/env python3
# coding: utf8

"""
Model of the unmix.io neuronal network learning object.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"


import os
import keras.utils
import numpy as np

from unmix.source.callbacks.callbacksfactory import CallbacksFactory
from unmix.source.configuration import Configuration
from unmix.source.data.datagenerator import DataGenerator
from unmix.source.data.dataloader import DataLoader
from unmix.source.helpers import console
from unmix.source.helpers import converter
from unmix.source.lossfunctions.lossfunctionfactory import LossFunctionFactory
from unmix.source.metrics.metricsfactory import MetricsFactory
from unmix.source.models.modelfactory import ModelFactory
from unmix.source.optimizers.optimizerfactory import OptimizerFactory
from unmix.source.pipeline.transformers.transformerfactory import TransformerFactory


class Engine:

    def __init__(self):
        self.callbacks = CallbacksFactory.build()
        optimizer = OptimizerFactory.build()
        loss_function = LossFunctionFactory.build()
        metrics = MetricsFactory.build()

        self.model = ModelFactory.build()
        self.model.compile(loss=loss_function,
                           optimizer=optimizer, metrics=metrics)
        self.model.summary()
        self.plot_model()
        console.debug("Model initialized with %d parameters." %
                      self.model.count_params())

        self.transformer = TransformerFactory.build()
        

    def plot_model(self):
        try:
            path = Configuration.get_path('environment.plot_folder')
            if path:
                name = Configuration.get('training.model').name
                file_name = os.path.join(
                    path, ('%s_%s-model.png' % (converter.get_timestamp(), name)))
                keras.utils.plot_model(self.model, file_name)
        except Exception as e:
            console.error("Error while plotting model: %s." % str(e))

    def train(self, epoch_start=0):
        training_songs, validation_songs = DataLoader.load()
        self.training_generator = DataGenerator(training_songs, self.transformer)
        self.validation_generator = DataGenerator(
            validation_songs, self.transformer)
        
        epoch_count = Configuration.get('training.epoch.count')
        history = self.model.fit_generator(
            generator=self.training_generator,
            validation_data=self.validation_generator,
            initial_epoch=epoch_start,
            epochs=epoch_start + epoch_count,
            shuffle=False,
            max_queue_size=10,
            verbose=Configuration.get('training.verbose'),
            callbacks=self.callbacks)
        self.save_weights()
        return history

    def predict(self, mix):
        length = self.transformer.calculate_items(mix.shape[1])
        predictions = []

        for i in range(length):
            input, transform_info = self.transformer.prepare_input(mix, i)
            predicted = self.model.predict(np.array([input]))[0]
            predicted = self.transformer.untransform_target(mix, predicted, i, transform_info)
            if predictions == []: # At this point, we know the shape of our predictions array - initialize
                shape = predicted.shape
                predictions = np.empty((shape[0], shape[1] * length), np.complex)
            predictions[:, shape[1] * i : shape[1] * (i+1)] = predicted
        
        return predictions

    def save_weights(self):
        path = os.path.join(Configuration.get_path(
            'environment.weights.folder'), Configuration.get('environment.weights.file'))
        console.info("Saved weights to: %s" % path)
        self.model.save_weights(path, overwrite=True)

    def load_weights(self, path=None):
        if not path:
            path = os.path.join(Configuration.get_path(
                'environment.weights.folder'), Configuration.get('environment.weights.file'))
        console.info("Load weights from: %s" % path)
        self.model.load_weights(path)
