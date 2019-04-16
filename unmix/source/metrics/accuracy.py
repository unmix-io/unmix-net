#!/usr/bin/env python3
# coding: utf8

"""
Builds metrics from configuration.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"

import librosa
import mir_eval
import numpy as np

from unmix.source.logging.logger import Logger
from unmix.source.data.song import Song
from unmix.source.prediction.mixprediciton import MixPrediciton


class Accuracy(object):

    def __init__(self, engine):
        self.engine = engine
        self.median_accuracies_vocals = []
        self.median_accuracies_instrumental = []

    def evaluate(self):
        accuracies_vocals = []
        accuracies_instrumental = []

        for song_file in self.engine.test_songs:
            try:
                song = Song(song_file)
                vocals = song.vocals.load().channels
                instrumental = song.instrumental.load().channels
                mix = vocals + instrumental

                prediction = MixPrediciton(self.engine)
                predicted_vocals, predicted_instrumental = prediction.run(
                    mix[0])

                accuracies_vocals.append(
                    self.__calculate_accuracy(vocals[0], predicted_vocals))
                accuracies_instrumental.append(self.__calculate_accuracy(
                    instrumental[0], predicted_instrumental))

            except Exception as e:
                Logger.error(
                    "Error while predicting song '%s': %s." % (song_file, str(e)))

        self.median_accuracies_vocals.append(
            self.__calculate_median(accuracies_vocals))
        self.median_accuracies_instrumental.append(
            self.__calculate_median(accuracies_instrumental))

        Logger.info("Median vocals: %s." %
                    (str(self.median_accuracies_vocals[-1])))
        Logger.info("Median instrumental: %s." %
                    (str(self.median_accuracies_instrumental[-1])))

    def __calculate_accuracy(self, original, predicted):
        audio_original = librosa.istft(original)
        audio_predicted = librosa.istft(predicted)

        result = mir_eval.separation.bss_eval_sources(
            audio_original, audio_predicted)
        entry = {
            'sdr': result[0][0],
            'sir': result[1][0],
            'sar': result[2][0],
            'perm': result[3][0]
        }
        return entry

    def __calculate_median(self, accuracies):
        return {
            'sdr': np.median(np.array([x['sdr'] for x in accuracies])),
            'sir': np.median(np.array([x['sir'] for x in accuracies])),
            'sar': np.median(np.array([x['sar'] for x in accuracies])),
            'perm': np.median(np.array([x['perm'] for x in accuracies]))
        }
