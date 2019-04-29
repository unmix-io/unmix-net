#!/usr/bin/env python3
# coding: utf8

"""
Builds callback checkpoint hooks for kera from configuration.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"


import os
import numpy as np
from keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger, EarlyStopping, ReduceLROnPlateau

from unmix.source.callbacks.errorvisualization import ErrorVisualization
from unmix.source.configuration import Configuration
from unmix.source.helpers import converter
from unmix.source.logging.logger import Logger


class CallbacksFactory(object):

    @staticmethod
    def build(build_validation_generator=None):
        configs = Configuration.get('training.callbacks', optional=False)
        callbacks = []
        if hasattr(configs, 'model_checkpoint'):
            callbacks.append(CallbacksFactory.model_checkpoint(
                configs.model_checkpoint))
        if hasattr(configs, 'tensorboard'):
            callbacks.append(CallbacksFactory.tensorboard(
                configs.tensorboard, build_validation_generator()))
        if hasattr(configs, 'csv_logger'):
            callbacks.append(CallbacksFactory.csv_logger(configs.csv_logger))
        if hasattr(configs, 'early_stopping'):
            callbacks.append(CallbacksFactory.early_stopping(
                configs.early_stopping))
        if hasattr(configs, 'reduce_learningrate'):
            callbacks.append(CallbacksFactory.reduce_learningrate(
                configs.reduce_learningrate))
        return callbacks

    @staticmethod
    def model_checkpoint(config):
        folder = Configuration.get_path("environment.weights.folder")
        path = os.path.join(folder, config.file_name)
        return ModelCheckpoint(filepath=path, monitor=config.monitor,
                               save_best_only=config.best_only,
                               save_weights_only=config.weights_only,
                               mode=config.mode, period=config.period, verbose=config.verbose)

    @staticmethod
    def tensorboard(config, validation_generator):
        path = os.path.join(Configuration.get("training.callbacks.tensorboard.folder", optional=False),
                            os.path.basename(Configuration.output_directory))
        return TensorBoardWrapper(validation_generator, log_dir=path,
                                  histogram_freq=config.histogram_freq,
                                  write_graph=config.write_graph,
                                  write_grads=config.write_grads,
                                  write_images=config.write_images,
                                  embeddings_freq=config.embeddings_freq,
                                  update_freq=config.update_freq,
                                  batch_size=Configuration.get("training.batch_size", default=8))

    @staticmethod
    def csv_logger(config):
        path = os.path.join(Configuration.output_directory, config.file_name)
        #open(path, 'a').close()
        return CSVLogger(filename=path, separator=config.separator, append=config.append)

    @staticmethod
    def early_stopping(config):
        return EarlyStopping(monitor=config.monitor,
                             min_delta=config.min_delta,
                             patience=config.patience,
                             verbose=config.verbose)

    @staticmethod
    def reduce_learningrate(config):
        return ReduceLROnPlateau(monitor=config.monitor, factor=config.factor,
                                 patience=config.patience, min_lr=config.min_learningrate)

    @staticmethod
    def error_visualization(bot):
        return ErrorVisualization(bot)  # TODO ???


class TensorBoardWrapper(TensorBoard):
    '''Sets the self.validation_data property for use with TensorBoard callback.'''

    def __init__(self, batch_gen, **kwargs):
        super().__init__(**kwargs)
        self.batch_gen = batch_gen  # The generator.

    def on_epoch_end(self, epoch, logs):
        # Fill in the `validation_data` property makes TensorBoard plot histograms etc.
        nb_steps = len(self.batch_gen)
        if(nb_steps == 0):
            Logger.debug(
                "No validation data provided; skip TensorBoard validation data")
            return
        input, output = None, None
        for s in range(nb_steps):
            ib, tb = self.batch_gen.__getitem__(s)
            if input is None and output is None:
                input = np.zeros(
                    (nb_steps * ib.shape[0], *ib.shape[1:]), dtype=np.float32)
                output = np.zeros(
                    (nb_steps * tb.shape[0], *tb.shape[1:]), dtype=np.uint8)
            input[s * ib.shape[0]:(s + 1) * ib.shape[0]] = ib
            output[s * tb.shape[0]:(s + 1) * tb.shape[0]] = tb
        self.validation_data = [input, output, np.ones(input.shape[0]), 0.0]
        self.batch_gen.on_epoch_end()
        return super().on_epoch_end(epoch, logs)
