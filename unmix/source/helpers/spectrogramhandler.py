#!/usr/bin/env python3
# coding: utf8

"""
Handles audio data and spectrograms.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"


import datetime
import os
import time
import librosa
from matplotlib.cm import get_cmap
import numpy as np
import skimage.io as io
import warnings

from unmix.source.configuration import Configuration
from unmix.source.helpers import reducer
from unmix.source.logging.logger import Logger


def to_audio(file, spectrogram):
    try:
        audio = np.array(librosa.istft(spectrogram))
        path = os.path.join(Configuration.get_path(
            'environment.temp_folder', optional=False), file)
        librosa.output.write_wav(path, audio, 11025, norm=False)
        Logger.info("Generated audio file from spectrogram: %s" % path)
    except:
        Logger.error("Error while saving spectrogram to image '%s'." % file)


def to_image(file, image):
    try:
        path = os.path.join(Configuration.get_path(
            'environment.plot_folder', optional=False), file)
        cm_hot = get_cmap('plasma')
        with warnings.catch_warnings():
            image = cm_hot(image)
            if len(image.shape) == 4:
                image = reducer.rflatter(image)
            warnings.simplefilter('ignore')
            io.imsave(path, image)
        Logger.info("Generated image: %s" % path)
    except:
        Logger.error("Error while saving spectrogram (with shape %s) to image '%s'." % (
            str(image.shape), file))

def remove_panning():
    """
    Takes a stereo file and removes the panning.
    """
    return