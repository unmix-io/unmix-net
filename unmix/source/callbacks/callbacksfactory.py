
"""
Builds callback checkpoint hooks for kera from configuration.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"


import os

from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau

from unmix.source.callbacks.errorvisualization import ErrorVisualization
from unmix.source.configuration import Configuration
from unmix.source.helpers import converter


class CallbacksFactory(object):

    @staticmethod
    def build():
        configs = Configuration.get('training.callbacks', False)
        callbacks = []
        if hasattr(configs, 'model_checkpoint'):
            callbacks.append(CallbacksFactory.model_checkpoint(configs.model_checkpoint))
        if hasattr(configs, 'early_stopping'):
            callbacks.append(CallbacksFactory.early_stopping(configs.early_stopping))
        if hasattr(configs, 'tensorboard'):
            callbacks.append(CallbacksFactory.tensorboard(configs.tensorboard))
        if hasattr(configs, 'reduce_learningrate'):
            callbacks.append(CallbacksFactory.reduce_learningrate(configs.reduce_learningrate))
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

    @staticmethod
    def tensorboard(config):
        path = Configuration.get_path("training.callbacks.tensorboard.folder")
        return TensorBoard(log_dir=path,
                           histogram_freq=config.histogram_freq,
                           write_graph=config.write_graph,
                           write_grads=config.write_grads,
                           write_images=config.write_images,
                           embeddings_freq=config.embeddings_freq,
                           update_freq=config.update_freq,
                           batch_size=Configuration.get("training.batch_size"))
