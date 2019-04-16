#!/usr/bin/env python3
# coding: utf8

"""
Model of a prediction of a song from a stream.
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

from unmix.source.prediction.prediction import Prediction
from unmix.source.configuration import Configuration
from unmix.source.data.track import Track
from unmix.source.exceptions.dataerror import DataError
from unmix.source.helpers import converter
from unmix.source.logging.logger import Logger


class FilePrediction(Prediction):

    def __init__(self, engine, sample_rate=22050, fft_window=1536):
        super().__init__(engine, sample_rate, fft_window)
        self.length = 0

    def run(self, file):
        'Predicts an audio file by loading the spectrogram and mixing the tracks.'
        audio, self.sample_rate_origin = librosa.load(
            file, mono=True, sr=self.sample_rate)
        self.mix = librosa.stft(audio, self.fft_window)

        Logger.info("Start predicting file: %s." % file)
        self.length = self.transformer.calculate_items(self.mix.shape[1])
        with progressbar.ProgressBar(max_value=self.length) as progbar:
            self.progressbar = progbar
            for i in range(self.length):
                input, transform_info = self.transformer.prepare_input(
                    self.mix, i)
                self.__predict_part(i, input, transform_info)

        self.__unpad()
        return self.vocals, self.instrumental
