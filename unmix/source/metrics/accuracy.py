#!/usr/bin/env python3
# coding: utf8

"""
Builds metrics from configuration.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"

import librosa
import mir_eval

from unmix.source.logging.logger import Logger
from unmix.source.data.song import Song


class Accuracy(object):

    def __init__(self, engine):
        self.engine = engine

    def evaluate(self):
        for song_file in self.engine.test_songs:
            try:
                song = Song(song_file)
                vocals = song.vocals.load().channels
                instrumentals = song.instrumental.load().channels
                mix = vocals + instrumentals

                predicted_vocals, predicted_instrumental = self.engine.predict(mix[0])

                audio_vocals = librosa.istft(vocals)
                audio_instrumentals = librosa.istft(instrumentals)
                audio_predicted_vocals = librosa.istft(predicted_vocals)
                audio_predicted_instrumentals = librosa.istft(predicted_instrumental)

                sdrv, sirv, sarv, permv = mir_eval.separation.bss_eval_sources(audio_vocals, audio_predicted_vocals)
                sdri, siri, sari, permi = mir_eval.separation.bss_eval_sources(audio_instrumentals, audio_predicted_instrumentals)
                Logger.info("mir_eval vocals: sdr=%s, sir=%s, sar=%s, perm=%s\n", str(sdrv), str(sirv), str(sarv), str(permv))
                Logger.info("mir_eval instrumentals: sdr=%s, sir=%s, sar=%s, perm=%s\n", str(sdri), str(siri), str(sari), str(permi))

            except Exception as e:
                print(
                    "Error while predicting song '%s': %s." % (song_file, str(e)))
