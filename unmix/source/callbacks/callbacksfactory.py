
"""
Builds callback checkpoint hooks for kera from configuration.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"


import os

from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping

from unmix.source.callbacks.errorvisualization import ErrorVisualization
from unmix.source.configuration import Configuration
from unmix.source.helpers import converter


class CallbacksFactory(object):

    @staticmethod
    def build():
        configs = Configuration.get('training.callbacks', False)
        callbacks = []
        if hasattr(configs, 'weights'):
            callbacks.append(CallbacksFactory.weights(configs.weights))
        if hasattr(configs, 'early_stopping'):
            callbacks.append(CallbacksFactory.early_stopping(configs.early_stopping))
        if hasattr(configs, 'tensorboard'):
            callbacks.append(CallbacksFactory.tensorboard(configs.tensorboard))
        return callbacks

    @staticmethod
    def weights(config):
        folder = Configuration.get_path("environment.weights.folder")
        file = os.path.join(folder, config.file_name)
        return ModelCheckpoint(filepath=file,
                               verbose=config.verbose,
                               save_best_only=config.best_only)

    @staticmethod
    def early_stopping(config):
        return EarlyStopping(min_delta=config.min_delta,
                             patience=config.patience,
                             verbose=config.verbose)

    @staticmethod
    def error_visualization(bot):
        return ErrorVisualization(bot)  # TODO ???

    @staticmethod
    def tensorboard(config):
        path = Configuration.get_path("environment.tensorboard_folder")
        log_dir = os.path.join(path, converter.get_timestamp())
        return TensorBoard(log_dir=log_dir,
                           write_images=config.write_images,
                           write_grads=config.write_grads,
                           histogram_freq=config.histogram_freq,
                           batch_size=Configuration.get("training.batch_size"))
