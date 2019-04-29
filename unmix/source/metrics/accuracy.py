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
        self.median_accuracies_mix = []
        self.fields = ['epoch', 'type', Accuracy.SDR,
                       Accuracy.SIR, Accuracy.SAR, Accuracy.PERM]
        self.file_vocals = self.__create_file("vocals")
        self.file_instrumental = self.__create_file("instrumental")
        self.file_mix = self.__create_file("mix")
        self.save_count = Configuration.get('collection.test_save_count', default=0)
        self.save_path = Configuration.build_path("predictions")

    def evaluate(self, epoch):
        accuracies_vocals = []
        accuracies_instrumental = []
        accuracies_mix = []
        i = 0
        for song_file in self.engine.test_songs:
            try:
                song = Song(song_file)
                vocals = song.vocals.load().channels
                instrumental = song.instrumental.load().channels
                mix = vocals + instrumental

                prediction = MixPrediciton(self.engine, sample_rate=Configuration.get('collection.sample_rate'))
                predicted_vocals, predicted_instrumental = prediction.run(
                    mix[0])
                if self.save_count > i:
                    prediction.save_vocals(song_file, folder=self.save_path)
                    prediction.save_instrumental(
                        song_file, folder=self.save_path)

                accuracies_vocals.append(
                    self.__calculate_accuracy_track(vocals[0], predicted_vocals))
                accuracies_instrumental.append(self.__calculate_accuracy_track(
                    instrumental[0], predicted_instrumental))
                accuracies_mix.append(
                    self.__calculate_accuracy_mix(vocals[0], instrumental[0], predicted_vocals, predicted_instrumental))
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
        mix_median = self.__calculate_median(
            accuracies_mix, 'mix', epoch)
        self.median_accuracies_mix.append(mix_median)

        self.__write_row(self.file_vocals, vocals_median)
        self.__write_row(self.file_instrumental, instrumental_median)
        self.__write_row(self.file_mix, mix_median)

        Logger.info("Median vocals: %s." %
                    (str(vocals_median)))
        Logger.info("Median instrumental: %s." %
                    (str(instrumental_median)))

    def __calculate_accuracy_track(self, original, predicted):
        result = mir_eval.separation.bss_eval_sources(
            librosa.istft(original), librosa.istft(predicted))
        entry = {
            Accuracy.SDR: result[0][0],
            Accuracy.SIR: result[1][0],
            Accuracy.SAR: result[2][0],
            Accuracy.PERM: result[3][0]
        }
        return entry

    def __calculate_accuracy_mix(self, original_vocals, orignal_instrumental, predicted_vocals, predicted_instrumental):
        result = mir_eval.separation.bss_eval_sources(
            np.array([librosa.istft(original_vocals),
                      librosa.istft(orignal_instrumental)]),
            np.array([librosa.istft(predicted_vocals), librosa.istft(predicted_instrumental)]))
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
            file_writer = csv.DictWriter(
                file, delimiter=';', fieldnames=self.fields)
            file_writer.writerow(data)
