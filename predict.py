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

from unmix.source.choppers.choppersfactory import ChoppersFactory
from unmix.source.configuration import Configuration
from unmix.source.helpers import console
from unmix.source.normalizers.normalizerfactory import NormalizerFactory
from unmix.source.engine import Engine


if __name__ == "__main__":
    global config

    parser = argparse.ArgumentParser(description="Executes a training session.")
    parser.add_argument('--configuration', default='./configurations/default.json',
                        type=str, help="Environment and training configuration.")
    parser.add_argument('--workingdir', default=os.getcwd(), 
                        type=str, help="Working directory (default: current directory).")
    parser.add_argument('--sample_rate', default=11025, 
                        type=str, help="Target sample rate which the model can process.")
    parser.add_argument('--fft_window', default=1536, 
                        type=str, help="FFT window size the model was trained on.")
    parser.add_argument('--file', default="test.mp3",
                        type=str, help="Input audio file to split vocals and instrumental.")
    parser.add_argument('--weights', default="",
                        type=str, help="Pretrained weights file (overwrites configuration).")

    args = parser.parse_args()

    config = Configuration.initialize(args.configuration, args.workingdir)

    normalizer = NormalizerFactory.build()

    console.info("Arguments: ", str(args))

    start = time.time()

    # Load song and create spectrogram with librosa
    audio, sample_rate = librosa.load(args.file, mono=True, sr=args.sample_rate)
    stft = librosa.stft(audio, args.fft_window)
    
    chopper = ChoppersFactory.build()[0]
    chops = chopper.chop(stft)
    chopshape = chops[0].shape

    engine = Engine()
    engine.load_weights(args.weights)

    predictions = np.empty((chopshape[0], chopshape[1] * len(chops)), dtype=np.complex)

    # Predict each chop
    for i,c in enumerate(chops):
        realimag = np.zeros((c.shape[0], c.shape[1], 2))
        realimag[:, :, 0] = np.real(c)
        realimag[:, :, 1] = np.imag(c)

        normalized, normalizer_info = normalizer.normalize(realimag)
        predicted = engine.predict(np.array([normalized]))[0]
        predicted = normalizer.denormalize(predicted, c, normalizer_info)
        predictions[:, chopshape[1] * i : (chopshape[1] * (i + 1))] = predicted

    # Convert back to wav audio file
    data = librosa.istft(predictions)
    audio = np.array(data)
    print('Output audio file: %s' % args.file + "_predicted.wav")
    librosa.output.write_wav(args.file + "_predicted.wav", audio, sample_rate, norm=False)

    end = time.time()

    console.info("Finished processing in %d [ms]." % (end - start))
