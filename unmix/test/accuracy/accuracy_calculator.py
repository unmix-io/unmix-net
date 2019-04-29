#!/usr/bin/env python3
# coding: utf8

"""
Predicts vocal and/or instrumental for a song.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"


import argparse
import glob
import numpy as np
import os
import time
import mir_eval
import librosa

from unmix.source.helpers import filehelper
from unmix.source.configuration import Configuration
from unmix.source.logging.logger import Logger

from unmix.source.data.song import Song

AUDIO_FILE_EXTENSIONS = ['.wav', '.mp3']


def __calculate_accuracy_mix(original_vocals, orignal_instrumental, predicted_vocals, predicted_instrumental):
    result = mir_eval.separation.bss_eval_sources(
        np.array([librosa.istft(original_vocals),
                  librosa.istft(orignal_instrumental)]),
        np.array([librosa.istft(predicted_vocals), librosa.istft(predicted_instrumental)]))
    print(result)


def __calculate_accuracy_track(original, predicted):
    result = mir_eval.separation.bss_eval_sources(
        librosa.istft(original), librosa.istft(predicted))
    print(result)

if __name__ == "__main__":
    global config

    parser = argparse.ArgumentParser(
        description="Executes a training session.")
    parser.add_argument('--run_folder', default='',
                        type=str, help="General training input folder (overwrites other parameters)")
    parser.add_argument('--configuration', default='../../../configurations/default-mask.jsonc',
                        type=str, help="Environment and training configuration.")
    parser.add_argument('--weights', default='',
                        type=str, help="Pretrained weights file (overwrites configuration).")
    parser.add_argument('--workingdir', default='../../../',
                        type=str, help="Working directory (default: current directory).")
    parser.add_argument('--sample_rate', default=11025,
                        type=str, help="Target sample rate which the model can process.")
    parser.add_argument('--fft_window', default=1536,
                        type=str, help="FFT window size the model was trained on.")
    parser.add_argument('--song', default='S:\\data-muellrap\\4_training\\fft-window=1536_sample-rate=11025_channels=1-mono\musdb18\\A Classic Education - NightOwl',
                        type=str, help="Input audio file to split vocals and instrumental.")
    parser.add_argument('--songs', default='',
                        type=str, help="Input folder containing audio files to split vocals and instrumental.")
    parser.add_argument('--youtube', default='', type=str,
                        help="Audio stream from a youtube video.")

    args = parser.parse_args()
    start = time.time()

    if args.run_folder:
        args.workingdir = filehelper.build_abspath(args.run_folder, os.getcwd())
        args.configuration = os.path.join(args.workingdir, 'configuration.jsonc')
        args.weights = filehelper.get_latest(os.path.join(args.workingdir, 'weights'), '*weights*.h5')

    Configuration.initialize(args.configuration, args.workingdir, False)
    Logger.initialize(False)

    if args.run_folder:
        prediction_folder = Configuration.build_path('predictions')
    else:
        prediction_folder = ''

    if os.path.isdir(args.song):
        args.songs = args.song
        args.song = ''

    Logger.info("Arguments: ", str(args))

    song = Song(args.songs)
    vocals = song.vocals.load().channels
    instrumentals = song.instrumental.load().channels

    __calculate_accuracy_mix(vocals[0], instrumentals[0], vocals[0], instrumentals[0])
    __calculate_accuracy_track(vocals[0], vocals[0])

