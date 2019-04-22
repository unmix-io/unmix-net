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

    SDR = 'sdr'
    SIR = 'sir'
    SAR = 'sar'
    PERM = 'perm'

    def __init__(self, engine):
        self.engine = engine
        self.median_accuracies_vocals = []
        self.median_accuracies_instrumental = []
        self.fields = ['epoch', 'type', Accuracy.SDR, Accuracy.SIR, Accuracy.SAR, Accuracy.PERM]
        self.file_vocals = self.__create_file("vocals")
        self.file_instrumental = self.__create_file("instrumental")
        self.save_count = Configuration.get('collection.test_save_count')
        self.save_path = Configuration.build_path("predictions")

    def evaluate(self, epoch):
        accuracies_vocals = []
        accuracies_instrumental = []
        i = 0
        for song_file in self.engine.test_songs:
            try:
                song = Song(song_file)
                vocals = song.vocals.load().channels
                instrumental = song.instrumental.load().channels
                mix = vocals + instrumental

                prediction = MixPrediciton(self.engine)
                predicted_vocals, predicted_instrumental = prediction.run(
                    mix[0])
                if self.save_count and self.save_count > i:
                    prediction.save_vocals(song_file, folder=self.save_path)
                    prediction.save_instrumental(song_file, folder=self.save_path)

                accuracies_vocals.append(
                    self.__calculate_accuracy(vocals[0], predicted_vocals))
                accuracies_instrumental.append(self.__calculate_accuracy(
                    instrumental[0], predicted_instrumental))
                i += 1
            except Exception as e:
                Logger.error(
                    "Error while predicting song '%s': %s." % (song_file, str(e)))

        vocals_median = self.__calculate_median(
            accuracies_vocals, 'vocals', epoch)
        self.median_accuracies_vocals.append(vocals_median)
        instrumental_median = self.__calculate_median(
            accuracies_instrumental, 'instrumental', epoch)
        self.median_accuracies_instrumental.append(instrumental_median)

        self.__write_row(self.file_vocals, vocals_median)
        self.__write_row(self.file_instrumental, instrumental_median)

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
            Accuracy.SDR: result[0][0],
            Accuracy.SIR: result[1][0],
            Accuracy.SAR: result[2][0],
            Accuracy.PERM: result[3][0]
        }
        return entry

    def __calculate_median(self, accuracies, type, epochnr):
        return {
            'epoch': epochnr,
            'type': type,
            Accuracy.SDR: np.median(np.array([x[Accuracy.SDR] for x in accuracies])),
            Accuracy.SIR: np.median(np.array([x[Accuracy.SIR] for x in accuracies])),
            Accuracy.SAR: np.median(np.array([x[Accuracy.SAR] for x in accuracies])),
            Accuracy.PERM: np.median(
                np.array([x[Accuracy.PERM] for x in accuracies]))
        }

    def __create_file(self, type):
        path = os.path.join(Configuration.output_directory,
                            'accuracy_%s.csv' % type)
        with open(path, mode='w', newline='') as file:
            file_writer = csv.DictWriter(
                file, delimiter=';', fieldnames=self.fields)
            file_writer.writeheader()
        return path

    def __write_row(self, path, data):
        with open(path, mode='a', newline='') as file:
            file_writer = csv.DictWriter(file, delimiter=';', fieldnames=self.fields)
            file_writer.writerow(data)
