#!/usr/bin/env python3
# coding: utf8

"""
Model of a prediction of a song from a mix.
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
from unmix.source.helpers import spectrogramhandler


class MixPrediction(Prediction):

    def __init__(self, engine, sample_rate=22050, fft_window=1536):
        super().__init__(engine, sample_rate, fft_window)
    
    def run(self, mix, stereo=False, remove_panning=False):
        'Predicts an audio file mix.'
        self.mix = mix
        if remove_panning:
            self.mix = np.mean(spectrogramhandler.remove_panning(self.mix), axis=0)
        Logger.info("Start predicting mix.")
        self.length = self.transformer.calculate_items(self.mix[0].shape[1])
        with progressbar.ProgressBar(max_value=self.length) as progbar:
            self.progressbar = progbar
            for i in range(self.length):
                input = self.transformer.prepare_input(
                    self.mix, i)
                self.predict_part(i, input)
                self.progressbar.update(self.progress)

        self.unpad()
        return self.vocals, self.instrumental
