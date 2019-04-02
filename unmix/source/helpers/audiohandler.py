#!/usr/bin/env python3
# coding: utf8

"""
Handles audio data.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"


import datetime
import os
import time
import librosa
import numpy as np

from unmix.source.configuration import Configuration


def save_spectrogram(file, spectrogram):
    audio = np.array(librosa.istft(generate_stft(spectrogram)))
    path = os.path.join(Configuration.get_path(
        'environment.temp_folder'), file)
    librosa.output.write_wav(path, audio, 11025, norm=False)


def generate_stft(spectrogram):
    real_part = spectrogram[:, :, 0]
    imag_part = spectrogram[:, :, 1]
    stft = real_part + imag_part * 1j
    return stft
