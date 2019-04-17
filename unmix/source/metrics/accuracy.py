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
import os
import csv

from unmix.source.logging.logger import Logger
from unmix.source.data.song import Song
from unmix.source.prediction.mixprediciton import MixPrediciton
from unmix.source.configuration import Configuration


class Accuracy(object):

    def __init__(self, engine):
        self.engine = engine
        self.median_accuracies_vocals = []
        self.median_accuracies_instrumental = []
        self.SDR = 'sdr'
        self.SIR = 'sir'
        self.SAR = 'sar'
        self.PERM = 'perm'

    def evaluate(self, epochnr):
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

        vocals_median = self.__calculate_median(accuracies_vocals, 'vocals', epochnr)
        self.median_accuracies_vocals.append(vocals_median)
        instrumental_median = self.__calculate_median(accuracies_instrumental, 'instrumental', epochnr)
        self.median_accuracies_instrumental.append(instrumental_median)

        csv_exists = os.path.isfile(os.path.join(Configuration.output_directory, 'accuracy.csv'))
        with open(os.path.join(Configuration.output_directory, 'accuracy.csv'), mode='a', newline='') as accuracy_file:
            fieldnames = ['epoch', 'type', self.SDR, self.SIR, self.SAR, self.PERM]
            file_writer = csv.DictWriter(accuracy_file, delimiter=';', fieldnames=fieldnames)
            if not csv_exists:
                file_writer.writeheader()
            file_writer.writerow(vocals_median)
            file_writer.writerow(instrumental_median)

        Logger.info("Median vocals: %s." %
                    (str(vocals_median)))
        Logger.info("Median instrumental: %s." %
                    (str(instrumental_median)))

    def __calculate_accuracy(self, original, predicted):
        audio_original = librosa.istft(original)
        audio_predicted = librosa.istft(predicted)

        result = mir_eval.separation.bss_eval_sources(
            audio_original, audio_predicted)
        entry = {
            self.SDR: result[0][0],
            self.SIR: result[1][0],
            self.SAR: result[2][0],
            self.PERM: result[3][0]
        }
        return entry

    def __calculate_median(self, accuracies, type, epochnr):
        return {
            'epoch': epochnr,
            'type': type,
            self.SDR: np.median(np.array([x[self.SDR] for x in accuracies])),
            self.SIR: np.median(np.array([x[self.SIR] for x in accuracies])),
            self.SAR: np.median(np.array([x[self.SAR] for x in accuracies])),
            self.PERM: np.median(np.array([x[self.PERM] for x in accuracies]))
        }
