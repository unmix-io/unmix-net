#!/usr/bin/env python3
# coding: utf8

"""
Model of a prediction of a new song.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"


import glob
import h5py
import os
import math
import numpy as np
import progressbar
import librosa

from unmix.source.configuration import Configuration
from unmix.source.data.track import Track
from unmix.source.exceptions.dataerror import DataError
from unmix.source.helpers import converter
from unmix.source.logging.logger import Logger


class Prediction(object):

    def __init__(self, engine, sample_rate=22050, fft_window=1536, stereo=False):
        self.model = engine.model
        self.transformer = engine.transformer
        self.graph = engine.graph
        self.sample_rate = sample_rate
        self.sample_rate_origin = 0
        self.fft_window = fft_window
        self.stereo = stereo
        self.mix = np.empty(0)
        self.vocals = np.empty(0)
        self.instrumental = np.empty(0)
        self.progress = 0
        self.initialized = False
        self.length = 0

    def save(self, file, folder='', extension='wav'):
        'Saves the predicted vocals and instrumental track to separate audio files.'
        return self.save_vocals(file, folder, extension), self.save_instrumental(file, folder, extension)

    def save_vocals(self, file, folder='', extension='wav'):
        'Saves the predicted vocals track to an audio file.'
        return self.__save(self.vocals, 'vocals', file, folder, extension)

    def save_instrumental(self, file, folder='', extension='wav'):
        'Saves the predicted instrumental track to an audio file.'
        return self.__save(self.instrumental, 'instrumental', file, folder, extension)

    def __save(self, prediction, type, file, folder='', extension='wav'):
        name = os.path.splitext(os.path.basename(file))[0]
        file_name = converter.get_timestamp() + "_" + name + \
            '_predicted_%s.%s' % (type, extension)
        if folder:
            output_file = os.path.join(
                folder, file_name)
        else:
            output_file = os.path.join(os.path.dirname(file), file_name)

        if self.stereo:
            track = np.array([librosa.istft(channel)
                              for channel in prediction])
        else:
            track = librosa.istft(prediction[0])
        librosa.output.write_wav(
            output_file, track, self.sample_rate, norm=False)
        Logger.info("Output prediction file: %s" % output_file)
        return output_file

    def predict_part(self, i, part):
        with self.graph.as_default():
            predicted = self.model.predict(np.array([part]))[0]
        predicted_vocals, predicted_instrumental = \
            self.transformer.untransform_target(
                self.mix, predicted, i)
        if not self.initialized:
            self.__init_shapes(predicted_vocals.shape)

        self.__expand_track(predicted_vocals, self.vocals, i)
        self.__expand_track(predicted_instrumental, self.instrumental, i)
        self.progress += 1

    def __expand_track(self, prediction, track, i):
        # Transformers can have their own expander - use that if available
        if(getattr(self.transformer, "expand_track", None)):
            self.transformer.expand_track(prediction, track, i)
        else:
            left = prediction.shape[2] * i
            right = prediction.shape[2] * (i+1)
            size = track.shape[2]
            if size < right:
                track = np.append(track, np.zeros(
                    (track.shape[0], track.shape[1], right - size)), axis=2)
            track[:, :, left:right] = prediction

    def __init_shapes(self, shape):
        self.vocals = np.zeros(
            (shape[0], shape[1], self.transformer.step * self.length), np.complex)
        self.instrumental = np.zeros_like(self.vocals)
        self.initialized = True

    def unpad(self):
        mix_length = self.mix[0].shape[1]
        offset = self.transformer.size // 2
        self.vocals = self.vocals[:, :, offset:offset + mix_length]
        self.instrumental = self.instrumental[:, :, offset:offset + mix_length]
