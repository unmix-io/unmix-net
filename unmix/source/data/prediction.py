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
import numpy as np
import progressbar

from unmix.source.data.track import Track
from unmix.source.exceptions.dataerror import DataError


class Prediction(object):

    def __init__(self, mix, model, transformer):
        self.mix = mix
        self.model = model
        self.transformer = transformer
        self.length = transformer.calculate_items(mix.shape[1])
        self.vocals = []
        self.instrumental = []
        self.progress = 0
        self.initialized = False

    def run(self):
        with progressbar.ProgressBar(max_value=self.length) as progbar:
            self.progressbar = progbar
            [self.predict_part(i) for i in range(self.length)]

    def predict_part(self, i):
        input, transform_info = self.transformer.prepare_input(self.mix, i)
        predicted = self.model.predict(np.array([input]))[0]
        predicted_vocals, predicted_instrumental = \
            self.transformer.untransform_target(
                self.mix, predicted, i, transform_info)
        if not self.initialized:
            self.init_shapes(predicted_vocals.shape)

        self.vocals[:,
                    predicted_vocals.shape[1] * i:
                    predicted_vocals.shape[1] * (i+1)] = predicted_vocals
        self.instrumental[:,
                          predicted_instrumental.shape[1] * i:
                          predicted_instrumental.shape[1] * (i+1)] = predicted_instrumental
        self.progress += 1
        self.progressbar.update(self.progress)

    def init_shapes(self, shape):
        self.vocals = np.empty(
            (shape[0], shape[1] * self.length), np.complex)
        self.instrumental = np.empty_like(self.vocals)
        self.step = self.vocals.shape[1]
        self.initialized = True
