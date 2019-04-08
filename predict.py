#!/usr/bin/env python3
# coding: utf8

"""
Predicts vocal and/or instrumental for a song.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"


import argparse
import librosa
import glob
import numpy as np
import os
import time

from unmix.source.pipeline.transformers.transformerfactory import TransformerFactory
from unmix.source.configuration import Configuration
from unmix.source.logging.logger import Logger
from unmix.source.engine import Engine


AUDIO_FILE_EXTENSIONS = ['.wav', '.mp3']

if __name__ == "__main__":
    global config

    parser = argparse.ArgumentParser(description="Executes a training session.")
    parser.add_argument('--run_folder', default='G:\\home-flurydav\\scripts\\runs\\20190408-145844',
                        type=str, help="General training input folder (overwrites other parameters)")
    parser.add_argument('--configuration', default='',
                        type=str, help="Environment and training configuration.")
    parser.add_argument('--weights', default="",
                        type=str, help="Pretrained weights file (overwrites configuration).")
    parser.add_argument('--workingdir', default=os.getcwd(), 
                        type=str, help="Working directory (default: current directory).")
    parser.add_argument('--sample_rate', default=11025, 
                        type=str, help="Target sample rate which the model can process.")
    parser.add_argument('--fft_window', default=1536, 
                        type=str, help="FFT window size the model was trained on.")
    parser.add_argument('--song', default='',
                        type=str, help="Input audio file to split vocals and instrumental.")
    parser.add_argument('--songs', default='D:\\Data\\unmix.io\\songs',
                        type=str, help="Input folder containing audio files to split vocals and instrumental.")

    args = parser.parse_args()
    start = time.time()

    if args.run_folder:
        args.workingdir = Configuration.build_abspath(args.run_folder, os.getcwd())
        args.configuration = os.path.join(args.workingdir, 'configuration.jsonc')
        args.weights = os.path.join(args.workingdir, 'weights', 'weights.h5')
    
    Configuration.initialize(args.configuration, args.workingdir, False)
    Logger.initialize(False)

    Logger.info("Arguments: ", str(args))

    prediction_folder = Configuration.build_path('predictions')

    song_files = []
    if args.song:
        song_files = song_files.append(args.song)
    if args.songs:
        for file in glob.iglob(Configuration.build_abspath(args.songs, os.getcwd()) + '**/*', recursive=True):
            extension = os.path.splitext(file)[1].lower()
            if extension in AUDIO_FILE_EXTENSIONS:
                song_files.append(file)

    Logger.info("Found %d songs to predict." % len(song_files))
        
    engine = Engine()
    engine.load_weights(args.weights)

    for song_file in song_files:
        # Load song and create spectrogram with librosa
        audio, sample_rate = librosa.load(song_file, mono=True, sr=args.sample_rate)
        mix = librosa.stft(audio, args.fft_window)
        predicted_vocals = engine.predict(mix)

        # Convert back to wav audio file
        data = librosa.istft(predicted_vocals)
        audio = np.array(data)

        output_file = os.path.join(prediction_folder, os.path.basename(song_file) + "_predicted.wav")
        Logger.info("Output prediction file: %s." % output_file)
        librosa.output.write_wav(output_file, audio, sample_rate, norm=False)

    end = time.time()

    Logger.info("Finished processing in %d [s]." % (end - start))
