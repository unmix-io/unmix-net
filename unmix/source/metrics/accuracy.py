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


                result = mir_eval.separation.bss_eval_sources(audio_instrumentals, audio_predicted_instrumentals)
                entry = {
                    sdr: result[0][0],
                    sir: result[1][0]
                }
                self.accuracy_vocals.append(entry)
                Logger.info("mir_eval vocals: %s" % (str(entry))


                self.accuracy_vocals = [np.append(self.accuracy_vocals[i], result[0]) for i, result in enumerate(
                    mir_eval.separation.bss_eval_sources(audio_vocals, audio_predicted_vocals))]
                [self.accuracy_vocals[i] = np.append(self.accuracy_instrumental[i], result[0]) for i, result in enumerate(mir_eval.separation.bss_eval_sources(audio_instrumentals, audio_predicted_instrumentals))]

                Logger.info("mir_eval vocals: sdr=%s, sir=%s, sar=%s, perm=%s" % (str(self.accuracy_vocals[0][len(self.accuracy_vocals[0])-1]), str(self.accuracy_vocals[1][len(
                    self.accuracy_vocals[1])-1]), str(self.accuracy_vocals[2][len(self.accuracy_vocals[2])-1]), str(self.accuracy_vocals[3][len(self.accuracy_vocals[3])-1])))
                Logger.info("mir_eval instrumentals: sdr=%s, sir=%s, sar=%s, perm=%s" % (
                    str(self.accuracy_instrumental[0][len(
                        self.accuracy_instrumental[0]) - 1]),
                    str(self.accuracy_instrumental[1][len(
                        self.accuracy_instrumental[1]) - 1]),
                    str(self.accuracy_instrumental[2][len(
                        self.accuracy_instrumental[2]) - 1]),
                    str(self.accuracy_instrumental[3][len(self.accuracy_instrumental[3]) - 1])))

            except Exception as e:
                Logger.error(
                    "Error while predicting song '%s': %s." % (song_file, str(e)))

        # np.median(np.array(list(x['sdr'] for x in self.accuracy_vocals)))

        Logger.info("Median vocals: SDR = %s, SIR=%s, SAR=%s, PERM=%s" % (
            str(np.median(self.accuracy_vocals[0])),
            str(np.median(self.accuracy_vocals[1])),
            str(np.median(self.accuracy_vocals[2])),
            str(np.median(self.accuracy_vocals[3]))))
        Logger.info("Median instrumental: SDR = %s, SIR=%s, SAR=%s, PERM=%s" % (
            str(np.median(self.accuracy_instrumental[0])),
            str(np.median(self.accuracy_instrumental[1])),
            str(np.median(self.accuracy_instrumental[2])),
            str(np.median(self.accuracy_instrumental[3]))))
