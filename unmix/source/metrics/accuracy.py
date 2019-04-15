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

                # TODO Hacky until prediction unpadding is fixed -> Beautify
                predicted_vocals = predicted_vocals[:,32:-(32-(vocals.shape[-1] - int(vocals.shape[-1] / 64)*64))] 
                predicted_instrumental = predicted_instrumental[:,32:-(32-(instrumentals.shape[-1] - int(instrumentals.shape[-1] / 64)*64))]

                audio_vocals = librosa.istft(vocals[0])
                audio_instrumentals = librosa.istft(instrumentals[0])
                audio_predicted_vocals = librosa.istft(predicted_vocals)
                audio_predicted_instrumentals = librosa.istft(predicted_instrumental)

                sdrv, sirv, sarv, permv = mir_eval.separation.bss_eval_sources(audio_vocals, audio_predicted_vocals)
                sdri, siri, sari, permi = mir_eval.separation.bss_eval_sources(audio_instrumentals, audio_predicted_instrumentals)
                
                Logger.info("mir_eval vocals: sdr=%s, sir=%s, sar=%s, perm=%s" % (str(sdrv), str(sirv), str(sarv), str(permv)))
                Logger.info("mir_eval instrumentals: sdr=%s, sir=%s, sar=%s, perm=%s" % (str(sdri), str(siri), str(sari), str(permi)))

            except Exception as e:
                Logger.error(
                    "Error while predicting song '%s': %s." % (song_file, str(e)))
