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

from unmix.source.engine import Engine
from unmix.source.helpers import filehelper
from unmix.source.helpers import converter
from unmix.source.pipeline.transformers.transformerfactory import TransformerFactory
from unmix.source.configuration import Configuration
from unmix.source.logging.logger import Logger


AUDIO_FILE_EXTENSIONS = ['.wav', '.mp3']

if __name__ == "__main__":
    global config

    parser = argparse.ArgumentParser(description="Executes a training session.")
    parser.add_argument('--run_folder', default='',
                        type=str, help="General training input folder (overwrites other parameters)")
    parser.add_argument('--configuration', default='',
                        type=str, help="Environment and training configuration.")
    parser.add_argument('--weights', default='',
                        type=str, help="Pretrained weights file (overwrites configuration).")
    parser.add_argument('--workingdir', default=os.getcwd(), 
                        type=str, help="Working directory (default: current directory).")
    parser.add_argument('--sample_rate', default=11025,
                        type=str, help="Target sample rate which the model can process.")
    parser.add_argument('--fft_window', default=1536, 
                        type=str, help="FFT window size the model was trained on.")
    parser.add_argument('--song', default='',
                        type=str, help="Input audio file to split vocals and instrumental.")
    parser.add_argument('--songs', default='temp/songs',
                        type=str, help="Input folder containing audio files to split vocals and instrumental.")

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
            # Load song and create spectrogram with librosa
            audio, sample_rate = librosa.load(song_file, mono=True, sr=args.sample_rate)
            mix = librosa.stft(audio, args.fft_window)
            
            Logger.info("Start predicting song: %s." % song_file)
            predicted_vocals, predicted_instrumental = engine.predict(mix)

            # Convert back to wav audio file
            vocals = np.array(librosa.istft(predicted_vocals))
            instrumental = np.array(librosa.istft(predicted_instrumental))

            if prediction_folder:
                output_file = os.path.join(prediction_folder, converter.get_timestamp() + "_" + os.path.splitext(os.path.basename(song_file))[0])
            else:
                output_file = os.path.splitext(song_file)[0]

            librosa.output.write_wav(output_file + "_vocals.wav", vocals, sample_rate, norm=False)
            Logger.info("Output prediction file: %s_vocals.wav." % output_file)

            librosa.output.write_wav(output_file + "_instrumental.wav", instrumental, sample_rate, norm=False)
            Logger.info("Output prediction file: %s_instrumentals.wav." % output_file)
        except Exception as e:
            Logger.error("Error while predicting song '%s': %s." % (song_file, str(e)))

    end = time.time()

    Logger.info("Finished processing in %d [s]." % (end - start))
