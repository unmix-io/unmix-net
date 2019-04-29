#!/usr/bin/env python3
# coding: utf8

"""
Model of the unmix.io neuronal network learning object.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"


import os
import keras.utils
import tensorflow as tf
import numpy as np

from unmix.source.callbacks.callbacksfactory import CallbacksFactory
from unmix.source.configuration import Configuration
from unmix.source.data.datagenerator import DataGenerator
from unmix.source.data.dataloader import DataLoader
from unmix.source.logging.logger import Logger
from unmix.source.helpers import converter
from unmix.source.lossfunctions.lossfunctionfactory import LossFunctionFactory
from unmix.source.metrics.metricsfactory import MetricsFactory
from unmix.source.models.modelfactory import ModelFactory
from unmix.source.optimizers.optimizerfactory import OptimizerFactory
from unmix.source.pipeline.transformers.transformerfactory import TransformerFactory


class Engine:

    def __init__(self):

        self.model = ModelFactory.build()
        optimizer = OptimizerFactory.build()
        loss_function = LossFunctionFactory.build(self.model)
        metrics = MetricsFactory.build()
        
        self.model.compile(loss=loss_function,
                           optimizer=optimizer, metrics=metrics)
        self.model.summary(print_fn=Logger.info)
        
        Logger.debug("Model '%s' initialized with %d parameters." %
                     (Configuration.get('training.model.name'), self.model.count_params()))

        self.transformer = TransformerFactory.build()
        self.test_songs = None
        self.graph = tf.get_default_graph()

    def plot_model(self):
        try:
            path = Configuration.get_path('environment.plot_folder')
            if path:
                name = Configuration.get('training.model').name
                file_name = os.path.join(
                    path, ('%s_%s-model.png' % (converter.get_timestamp(), name)))
                keras.utils.plot_model(self.model, file_name)
        except Exception as e:
            Logger.warn("Error while plotting model: %s" % str(e))

    def train(self, epoch_start=0):
        training_songs, validation_songs, test_songs = DataLoader.load()
        self.training_generator = DataGenerator('training',
            self, training_songs, self.transformer, False)
        self.validation_generator = DataGenerator('validation',
            self, validation_songs, self.transformer, True)
        self.test_songs = test_songs

        build_validation_generator = lambda: DataGenerator('validation_tensorboard', self, validation_songs, self.transformer, False)
        # Pass a new data generator here because TensorBoard must have access to validation_data
        self.callbacks = CallbacksFactory.build(build_validation_generator)

        epoch_count = Configuration.get('training.epoch.count', optional=False)
        history = self.model.fit_generator(
            generator=self.training_generator,
            validation_data=self.validation_generator,
            initial_epoch=epoch_start,
            epochs=epoch_start + epoch_count,
            shuffle=False,
            max_queue_size=10,
            verbose=Configuration.get('training.verbose'),
            callbacks=self.callbacks)
        self.save()
        self.save_weights()
        return history
    
    def save(self):
        path = Configuration.get_path('environment.model_file', optional=False)
        Logger.info("Saved model and weights to: %s" % path)
        self.model.save(path, overwrite=True)
    
    def save_weights(self):
        path = os.path.join(Configuration.get_path(
            'environment.weights.folder', optional=False), Configuration.get('environment.weights.file', optional=False))
        Logger.info("Saved weights to: %s" % path)
        self.model.save_weights(path, overwrite=True)

    def load(self, path=None):
        if not path:
            path = Configuration.get_path('environment.model_file', optional=False)
        Logger.info("Load model and weights from: %s" % path)
        self.model.load(path)

    def load_weights(self, path=None):
        if not path:
            path = os.path.join(Configuration.get_path(
                'environment.weights.folder', optional=False), Configuration.get('environment.weights.file', optional=False))
        Logger.info("Load weights from: %s" % path)
        self.model.load_weights(path)
