#!/usr/bin/env python3
# coding: utf8

"""
Model of a prediction of a downloaded song from youtube.
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
from unmix.source.prediction.fileprediction import FilePrediction
from unmix.source.configuration import Configuration
from unmix.source.data.track import Track
from unmix.source.exceptions.dataerror import DataError
from unmix.source.helpers import converter
from unmix.source.logging.logger import Logger


class YoutTubePrediction(FilePrediction):

    def __init__(self, engine, sample_rate=22050, fft_window=1536):
        super().__init__(engine, sample_rate, fft_window)
        self.length = 1
        self.cursor = 0

    def run(self, link):
        'Predicts an download from youtube.'
        Logger.info("Start downloading youtube song: %s." % link)
        path, name = self.__download(link)
        super().run(os.path.join(path, name))
        return path, name

    def __download(self, link):        
        from pytube import YouTube
        yt = YouTube(link)
        yt.register_on_progress_callback(self.__youtube_stream_callback)
        audio = yt.streams.filter(subtype='mp4') \
                .order_by('resolution').desc().first() # only_audio=True is much slower
        self.length = audio.filesize
        # self.sample_rate_origin = int(audio.audio_sample_rate)
        path = Configuration.get_path('environment.temp_folder')

        with progressbar.ProgressBar(max_value=self.length) as progbar:
            self.progressbar = progbar
            audio.download(path)
        return path, audio.default_filename

    def __youtube_stream_callback(self, stream, chunk, file_handle, bytes_remaining):
        self.progressbar.update(self.length - bytes_remaining)