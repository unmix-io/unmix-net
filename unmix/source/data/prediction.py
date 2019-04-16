#!/usr/bin/env python3
# coding: utf8

"""
Model of a prediction of a new song.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"


import glob
import h5py
import os
import numpy as np
import progressbar
import librosa

from unmix.source.data.track import Track
from unmix.source.exceptions.dataerror import DataError
from unmix.source.helpers import converter
from unmix.source.logging.logger import Logger


class Prediction(object):

    def __init__(self, engine):
        self.model = engine.model
        self.transformer = engine.transformer
        self.graph = engine.graph
        self.vocals = []
        self.instrumental = []
        self.progress = 0
        self.initialized = False
        self.sample_rate = 22050

    def predict_file(self, file, sample_rate=22050, fft_window=1536):
        'Predicts an audio file by loading the spectrogram and mixing the tracks.'
        self.sample_rate = sample_rate
        audio, sample_rate = librosa.load(
            file, mono=True, sr=sample_rate)
        mix = librosa.stft(audio, fft_window)
        Logger.info("Start predicting file: %s." % file)
        return self.predict_mix(mix)

    def predict_mix(self, mix):
        'Predicts a mixed sftf soung representation.'
        self.mix = mix
        self.length = self.transformer.calculate_items(mix.shape[1])
        Logger.info("Start predicting mix.")
        with self.graph.as_default():
            self.__run()
        return self.vocals, self.instrumental

    def predict_youtube(self, link):
        'Predicts an audio stream from youtube.'
        from pytube import YouTube
        Logger.info("Start predicting stream from youtube link: %s." % link)
        yt = YouTube(link)
        yt.streams \
            .filter(progressive=True, only_audio=True) \
            .order_by('resolution') \
            .desc() \
            .first() \
            .download()
        yt.register_on_progress_callback(self.__youtube_stream_callback)
        return False

    def __youtube_stream_callback(self, stream, chunk, file_handle, bytes_remaining):
        print("Hello there.")

    def save_vocals(self, file, folder='', extension='wav'):
        'Saves the predicted instrumental track to an audio file'
        self.__save(self.vocals, 'vocals', file, folder, extension)

    def save_instrumental(self, file, folder='', extension='wav'):
        'Saves the predicted instrumental track to an audio file'
        self.__save(self.instrumental, 'instrumental', file, folder, extension)

    def __save(self, prediciton, type, file, folder='', extension='wav'):
        track = np.array(librosa.istft(prediciton))
        name = os.path.splitext(os.path.basename(file))[0]
        file_name = converter.get_timestamp() + "_" + name + \
            '_predicted_%s.%s' % (type, extension)
        if folder:
            output_file = os.path.join(
                folder, file_name)
        else:
            output_file = os.path.join(os.path.dirname(file), file_name)
        librosa.output.write_wav(
            output_file, track, self.sample_rate, norm=False)
        Logger.info("Output prediction file: %s" % output_file)

    def __run(self):
        with progressbar.ProgressBar(max_value=self.length) as progbar:
            self.progressbar = progbar
            [self.__predict_part(i) for i in range(self.length)]
            self.vocals = self.vocals[:, int(self.transformer.size/2):- (self.transformer.size - ((int(self.transformer.size/2) + self.mix.shape[1]) % self.transformer.size))]
            self.instrumental = self.instrumental[:, int(self.transformer.size/2):- (self.transformer.size - ((int(self.transformer.size/2) + self.mix.shape[1]) % self.transformer.size))]

    def __predict_part(self, i):
        input, transform_info = self.transformer.prepare_input(self.mix, i)
        predicted = self.model.predict(np.array([input]))[0]
        predicted_vocals, predicted_instrumental = \
            self.transformer.untransform_target(
                self.mix, predicted, i, transform_info)
        if not self.initialized:
            self.__init_shapes(predicted_vocals.shape)

        self.vocals[:,
                    predicted_vocals.shape[1] * i:
                    predicted_vocals.shape[1] * (i+1)] = predicted_vocals
        self.instrumental[:,
                          predicted_instrumental.shape[1] * i:
                          predicted_instrumental.shape[1] * (i+1)] = predicted_instrumental
        self.progress += 1
        self.progressbar.update(self.progress)

    def __init_shapes(self, shape):
        self.vocals = np.empty(
            (shape[0], shape[1] * self.length), np.complex)
        self.instrumental = np.empty_like(self.vocals)
        self.step = self.vocals.shape[1]
        self.initialized = True
