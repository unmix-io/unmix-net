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

    def evaluate(self):
        self.accuracy_vocals = []
        self.accuracy_instrumental = []

        for song_file in self.engine.test_songs:
            try:
                song = Song(song_file)
                vocals = song.vocals.load().channels
                instrumentals = song.instrumental.load().channels
                mix = vocals + instrumentals

                prediction = MixPrediciton(self.engine)
                predicted_vocals, predicted_instrumental = prediction.run(
                    mix[0])

                audio_vocals = librosa.istft(vocals[0])
                audio_instrumentals = librosa.istft(instrumentals[0])
                audio_predicted_vocals = librosa.istft(predicted_vocals)
                audio_predicted_instrumentals = librosa.istft(
                    predicted_instrumental)

                result_vocals = mir_eval.separation.bss_eval_sources(audio_vocals, audio_predicted_vocals)
                entry_vocals = {
                    'sdr': result_vocals[0][0],
                    'sir': result_vocals[1][0],
                    'sar': result_vocals[2][0],
                    'perm': result_vocals[3][0]
                }
                self.accuracy_vocals.append(entry_vocals)
                Logger.info("mir_eval vocals: %s" % (str(entry_vocals)))

                result_instrumental = mir_eval.separation.bss_eval_sources(audio_instrumentals, audio_predicted_instrumentals)
                entry_instrumental = {
                    'sdr': result_instrumental[0][0],
                    'sir': result_instrumental[1][0],
                    'sar': result_instrumental[2][0],
                    'perm': result_instrumental[3][0]
                }
                self.accuracy_instrumental.append(entry_instrumental)
                Logger.info("mir_eval vocals: %s" % (str(entry_instrumental)))

            except Exception as e:
                Logger.error(
                    "Error while predicting song '%s': %s." % (song_file, str(e)))

        medians_vocals = {
            'sdr': np.median(np.array(list(x['sdr'] for x in self.accuracy_vocals))),
            'sir': np.median(np.array(list(x['sir'] for x in self.accuracy_vocals))),
            'sar': np.median(np.array(list(x['sar'] for x in self.accuracy_vocals))),
            'perm': np.median(np.array(list(x['perm'] for x in self.accuracy_vocals))),
        }
        median_instrumental = {
            'sdr': np.median(np.array(list(x['sdr'] for x in self.accuracy_instrumental))),
            'sir': np.median(np.array(list(x['sir'] for x in self.accuracy_instrumental))),
            'sar': np.median(np.array(list(x['sar'] for x in self.accuracy_instrumental))),
            'perm': np.median(np.array(list(x['perm'] for x in self.accuracy_instrumental))),
        }

        Logger.info("Median vocals: %s" % (str(medians_vocals)))
        Logger.info("Median vocals: %s" % (str(median_instrumental)))
