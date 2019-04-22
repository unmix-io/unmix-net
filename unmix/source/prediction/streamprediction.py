#!/usr/bin/env python3
# coding: utf8

"""
Model of a prediction of a song from a stream.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"


import glob
import h5py
import os
import math
import numpy as np
import progressbar
import librosa

from unmix.source.prediction.prediction import Prediction
from unmix.source.configuration import Configuration
from unmix.source.data.track import Track
from unmix.source.exceptions.dataerror import DataError
from unmix.source.helpers import converter
from unmix.source.logging.logger import Logger


class StreamPrediction(Prediction):

    CHUNK_SIZE = 8192
    PREDICTION_SIZE = 128

    def __init__(self, engine, sample_rate=22050, fft_window=1536):
        super().__init__(engine, sample_rate, fft_window)
        self.length = 1
        self.cursor = 0

    def run(self, link):
        'Predicts an audio stream from youtube.'
        from pytube import YouTube
        Logger.info("Start predicting stream from youtube link: %s." % link)
        yt = YouTube(link)
        yt.register_on_progress_callback(self.__youtube_stream_callback)
        audio = yt.streams.filter(only_audio=True, subtype='mp4').first()
        self.sample_rate_origin = int(audio.audio_sample_rate)
        path = Configuration.get_path('environment.temp_folder')
        with progressbar.ProgressBar(max_value=math.ceil(int(audio.filesize) / (StreamPrediction.CHUNK_SIZE))) as progbar:
            self.progressbar = progbar
            audio.download(path)
        return False

    def __youtube_stream_callback(self, stream, chunk, file_handle, bytes_remaining):
        self.__predict_chunk(chunk)

    def __predict_chunk(self, chunk):
        try:
            data = np.nan_to_num(np.fromstring(chunk, dtype=np.float32))
            audio = librosa.resample(
                data, self.sample_rate_origin, self.sample_rate)
            mix = librosa.stft(audio, self.fft_window)
            if len(self.mix) <= 0:
                self.mix = mix
            else:
                self.mix = np.concatenate((self.mix, mix), axis=1)

            position = self.mix.shape[1] // StreamPrediction.PREDICTION_SIZE
            if position > self.cursor:
                input, transform_info = self.transformer.prepare_input(
                    self.mix[:, StreamPrediction.PREDICTION_SIZE*(position-1):StreamPrediction.PREDICTION_SIZE*position], 0)
                # TODO Move prediction to isolated thread?
                self.predict_part(self.progress, input, transform_info)
                self.cursor += 1
        except Exception as e:
            Logger.error("Error while predicting stream chunk: %s." % str(e))
        return None
