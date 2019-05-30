#!/usr/bin/env python3
# coding: utf8

"""
Handles audio data and spectrograms.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"


import os
import time
import librosa
import warnings
import datetime
import numpy as np
import skimage.io as io
from matplotlib.cm import get_cmap

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


def remove_panning(mix):
    """
    Takes a stereo file and removes the panning.
    """
    if len(mix) < 2:
        return mix
    left = mix[0]    
    right = mix[1]

    difference_left = __channel_difference_panning(left, right)
    left -= difference_left

    difference_right = __channel_difference_panning(right, left)
    right -= difference_right

    return [left, right]

def __channel_difference_panning(base, other, invert=True):
    amplitude = np.clip(np.abs(base) - np.abs(other), 0, None)
    phase = np.exp((np.angle(base) * (-1 if invert else 1)) * 1j)
    return amplitude * phase
