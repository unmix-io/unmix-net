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
from unmix.source.logging.logger import Logger


def spectrogram_to_audio(file, spectrogram):
    audio = np.array(librosa.istft(spectrogram))
    path = os.path.join(Configuration.get_path(
        'environment.temp_folder'), file)
    librosa.output.write_wav(path, audio, 11025, norm=False)
    Logger.info("Generated audio file from spectrogram: %s" % path)