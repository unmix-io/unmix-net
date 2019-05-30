#!/usr/bin/env python3
# coding: utf8

"""
Calculates the accuracy over a test set using the mir_eval bss evaluation.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"

import librosa
import mir_eval
import numpy as np
import os
import csv
import progressbar

from unmix.source.logging.logger import Logger
from unmix.source.data.song import Song
from unmix.source.prediction.mixprediction import MixPrediction
from unmix.source.configuration import Configuration


class Accuracy(object):

    SDR = 'sdr'
    SIR = 'sir'
    SAR = 'sar'
    PERM = 'perm'
    PREFIX_VOCALS = 'voc_'
    PREFIX_INSTRUMENTAL = 'inst_'

    def __init__(self, engine, name=''):
        self.engine = engine
        self.medians = []
        self.save_count = Configuration.get(
            'collection.test_save_count', default=0)
        self.save_path = Configuration.build_path("predictions")
        self.fields = ['epoch', 'type',
                       Accuracy.PREFIX_VOCALS + Accuracy.SDR,
                       Accuracy.PREFIX_VOCALS + Accuracy.SIR,
                       Accuracy.PREFIX_VOCALS + Accuracy.SAR,
                       Accuracy.PREFIX_INSTRUMENTAL + Accuracy.SDR,
                       Accuracy.PREFIX_INSTRUMENTAL + Accuracy.SIR,
                       Accuracy.PREFIX_INSTRUMENTAL + Accuracy.SAR]
        if name:
            self.file_name = 'accuracy_%s.csv' % name
        else:
            self.file_name = 'accuracy.csv'

    def evaluate(self, epoch, remove_panning=False):
        accuracies = []
        i = 0
        if not self.engine.test_songs:
            return
        with progressbar.ProgressBar(max_value=len(self.engine.test_songs)) as progbar:
            for song_file in self.engine.test_songs:
                try:
                    song = Song(song_file)
                    mix, vocals = song.load(
                        remove_panning=remove_panning, clean_up=False)
                    instrumental = song.instrumental.channels

                    prediction = MixPrediction(
                        self.engine, sample_rate=Configuration.get('collection.sample_rate'))
                    predicted_vocals, predicted_instrumental = prediction.run(
                        mix)
                    if self.save_count > i:
                        prediction.save_vocals(
                            song_file, folder=self.save_path)
                        prediction.save_instrumental(
                            song_file, folder=self.save_path)

                    accuracies.append(
                        self.__generate_accuracy(vocals, instrumental,
                                                 predicted_vocals, predicted_instrumental))
                except Exception as e:
                    Logger.error(
                        "Error while predicting song '%s': %s." % (song_file, str(e)))
                i += 1
                progbar.update(i)

        median = self.__calculate_median(
            accuracies, 'mix', epoch)
        self.medians.append(median)

        if not os.path.exists(os.path.join(Configuration.output_directory, self.file_name)):
            self.__create_file()

        self.__write_row(os.path.join(
            Configuration.output_directory, self.file_name), median)

        Logger.info("Updated accuracy results.")

    def __generate_accuracy(self, original_vocals, orignal_instrumental, predicted_vocals, predicted_instrumental):
        accuracies = [self.__calculate_accuracy(
            original_vocals[0], orignal_instrumental[0], predicted_vocals[0], predicted_instrumental[0])]
        if Configuration.get("collection.stereo", default=False):
            accuracies.append(self.__calculate_accuracy(
                original_vocals[1], orignal_instrumental[1], predicted_vocals[1], predicted_instrumental[1]))
        result = np.mean(accuracies, axis=0)
        entry = {
            Accuracy.PREFIX_VOCALS + Accuracy.SDR: result[0][0],
            Accuracy.PREFIX_VOCALS + Accuracy.SIR: result[1][0],
            Accuracy.PREFIX_VOCALS + Accuracy.SAR: result[2][0],
            Accuracy.PREFIX_INSTRUMENTAL + Accuracy.SDR: result[0][1],
            Accuracy.PREFIX_INSTRUMENTAL + Accuracy.SIR: result[1][1],
            Accuracy.PREFIX_INSTRUMENTAL + Accuracy.SAR: result[2][1],
        }
        return entry

    def __calculate_accuracy(self, original_vocals, orignal_instrumental, predicted_vocals, predicted_instrumental):
        return mir_eval.separation.bss_eval_sources(
            np.array([librosa.istft(original_vocals),
                      librosa.istft(orignal_instrumental)]),
            np.array([librosa.istft(predicted_vocals), librosa.istft(predicted_instrumental)]), compute_permutation=False)

    def __calculate_median(self, accuracies, type, epoch):
        return {
            'epoch': epoch,
            'type': type,
            Accuracy.PREFIX_VOCALS + Accuracy.SDR:
                np.median(
                    np.array([x[Accuracy.PREFIX_VOCALS + Accuracy.SDR] for x in accuracies])),
            Accuracy.PREFIX_VOCALS + Accuracy.SIR:
                np.median(
                    np.array([x[Accuracy.PREFIX_VOCALS + Accuracy.SIR] for x in accuracies])),
            Accuracy.PREFIX_VOCALS + Accuracy.SAR:
                np.median(
                    np.array([x[Accuracy.PREFIX_VOCALS + Accuracy.SAR] for x in accuracies])),
            Accuracy.PREFIX_INSTRUMENTAL + Accuracy.SDR:
                np.median(np.array(
                    [x[Accuracy.PREFIX_INSTRUMENTAL + Accuracy.SDR] for x in accuracies])),
            Accuracy.PREFIX_INSTRUMENTAL + Accuracy.SIR:
                np.median(np.array(
                    [x[Accuracy.PREFIX_INSTRUMENTAL + Accuracy.SIR] for x in accuracies])),
            Accuracy.PREFIX_INSTRUMENTAL + Accuracy.SAR:
                np.median(np.array(
                    [x[Accuracy.PREFIX_INSTRUMENTAL + Accuracy.SAR] for x in accuracies])),
        }

    def __create_file(self):
        path = os.path.join(Configuration.output_directory, self.file_name)
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
