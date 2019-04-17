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

from unmix.source.engine import Engine
from unmix.source.helpers import filehelper
from unmix.source.helpers import converter
from unmix.source.prediction.fileprediction import FilePrediction
from unmix.source.prediction.streamprediction import StreamPrediction
from unmix.source.pipeline.transformers.transformerfactory import TransformerFactory
from unmix.source.configuration import Configuration
from unmix.source.logging.logger import Logger


AUDIO_FILE_EXTENSIONS = ['.wav', '.mp3']

if __name__ == "__main__":
    global config

    parser = argparse.ArgumentParser(
        description="Executes a training session.")
    parser.add_argument('--run_folder', default='./runs/20190417-105759',
                        type=str, help="General training input folder (overwrites other parameters)")
    parser.add_argument('--configuration', default='',
                        type=str, help="Environment and training configuration.")
    parser.add_argument('--weights', default='',
                        type=str, help="Pretrained weights file (overwrites configuration).")
    parser.add_argument('--workingdir', default=os.getcwd(),
                        type=str, help="Working directory (default: current directory).")
    parser.add_argument('--sample_rate', default=22050,
                        type=str, help="Target sample rate which the model can process.")
    parser.add_argument('--fft_window', default=1536,
                        type=str, help="FFT window size the model was trained on.")
    parser.add_argument('--song', default='',
                        type=str, help="Input audio file to split vocals and instrumental.")
    parser.add_argument('--songs', default='',
                        type=str, help="Input folder containing audio files to split vocals and instrumental.")
    parser.add_argument('--youtube', default='', type=str,
                        help="Audio stream from a youtube video.")

    args = parser.parse_args()
    start = time.time()

    if args.run_folder:
        args.workingdir = filehelper.build_abspath(
            args.run_folder, os.getcwd())
        args.configuration = os.path.join(
            args.workingdir, 'configuration.jsonc')
        args.weights = filehelper.get_latest(os.path.join(
            args.workingdir, 'weights'), '*weights*.h5')

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

    song_files = []
    if args.song:
        song_files = [args.song]
    if args.songs:
        for file in glob.iglob(filehelper.build_abspath(args.songs, os.getcwd()) + '**/*', recursive=True):
            extension = os.path.splitext(file)[1].lower()
            if extension in AUDIO_FILE_EXTENSIONS:
                song_files.append(file)

    Logger.info("Found %d songs to predict." % len(song_files))

    engine = Engine()
    engine.load_weights(args.weights)

    for song_file in song_files:
        try:
            prediction = FilePrediction(engine, sample_rate=args.sample_rate, fft_window=args.fft_window)
            prediction.run(song_file)
            prediction.save_vocals(song_file, prediction_folder)
            prediction.save_instrumental(song_file, prediction_folder)
        except Exception as e:
            Logger.error(
                "Error while predicting song '%s': %s." % (song_file, str(e)))

    if args.youtube:
        prediction = StreamPrediction(engine, sample_rate=args.sample_rate, fft_window=args.fft_window)
        prediction.run(args.youtube)

    end = time.time()
    Logger.info("Finished processing in %d [s]." % (end - start))
