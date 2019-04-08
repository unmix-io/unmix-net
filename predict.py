#!/usr/bin/env python3
# coding: utf8

"""
Predicts vocal and/or instrumental for a song.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"


import argparse
import librosa
import numpy as np
import os
import time

from unmix.source.pipeline.transformers.transformerfactory import TransformerFactory
from unmix.source.configuration import Configuration
from unmix.source.helpers import console
from unmix.source.engine import Engine


if __name__ == "__main__":
    global config

    parser = argparse.ArgumentParser(description="Executes a training session.")
    parser.add_argument('--configuration', default='',
                        type=str, help="Environment and training configuration.")
    parser.add_argument('--workingdir', default=os.getcwd(), 
                        type=str, help="Working directory (default: current directory).")
    parser.add_argument('--sample_rate', default=11025, 
                        type=str, help="Target sample rate which the model can process.")
    parser.add_argument('--fft_window', default=1536, 
                        type=str, help="FFT window size the model was trained on.")
    parser.add_argument('--file', default="./temp/test.mp3",
                        type=str, help="Input audio file to split vocals and instrumental.")
    parser.add_argument('--weights', default="",
                        type=str, help="Pretrained weights file (overwrites configuration).")

    args = parser.parse_args()
    console.info("Arguments: ", str(args))
    start = time.time()

    Configuration.initialize(args.configuration, args.workingdir, False)

    # Load song and create spectrogram with librosa
    audio, sample_rate = librosa.load(args.file, mono=True, sr=args.sample_rate)
    mix = librosa.stft(audio, args.fft_window)
    
    engine = Engine()
    engine.load_weights(args.weights)
    predicted_vocals = engine.predict(mix)

    # Convert back to wav audio file
    data = librosa.istft(predicted_vocals)
    audio = np.array(data)
    print('Output audio file: %s' % args.file + "_predicted.wav")
    librosa.output.write_wav(args.file + "_predicted.wav", audio, sample_rate, norm=False)

    end = time.time()

    console.info("Finished processing in %d [s]." % (end - start))
