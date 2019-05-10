#!/usr/bin/env python3
# coding: utf8

"""
Measures the accuracy of a training run.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"


import argparse
import glob
import os
import time

from unmix.source.engine import Engine
from unmix.source.helpers import filehelper
from unmix.source.metrics.accuracy import Accuracy
from unmix.source.configuration import Configuration
from unmix.source.logging.logger import Logger
from unmix.source.data.song import Song
from unmix.source.data.dataloader import DataLoader


AUDIO_FILE_EXTENSIONS = ['.wav', '.mp3']

if __name__ == "__main__":
    global config

    parser = argparse.ArgumentParser(
        description="Executes a training session.")
    parser.add_argument('--run_folder', default='',
                        type=str, help="General training input folder.")
    parser.add_argument('--fft_window', default=1536,
                        type=str, help="FFT window size the model was trained on.")
    parser.add_argument('--data_path', default='',
                        type=str, help="Input folder containing audio files to split vocals and instrumental.")

    args = parser.parse_args()
    Logger.info("Arguments: ", str(args))
    start = time.time()

    workingdir = filehelper.build_abspath(args.run_folder, os.getcwd())
    configuration = os.path.join(workingdir, 'configuration.jsonc')
    weights = filehelper.get_latest(os.path.join(
        workingdir, 'weights'), '*weights*.h5')

    Configuration.initialize(configuration, workingdir, False)
    Logger.initialize(False)

    engine = Engine()
    engine.load_weights(weights)
    training_songs, validation_songs, test_songs = DataLoader.load()
    engine.accuracy = Accuracy(engine)
    engine.test_songs = test_songs

    if args.data_path:
        engine.test_songs = [os.path.dirname(file) for file in glob.iglob(
            os.path.join(args.data_path, '**', '%s*.h5' % Song.PREFIX_VOCALS), recursive=True)]
    Logger.info("Found %d songs to measure accuracy." % len(engine.test_songs))

    engine.accuracy.evaluate("measure")

    end = time.time()
    Logger.info("Finished processing in %d [s]." % (end - start))
