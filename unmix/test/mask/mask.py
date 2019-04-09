import numpy as np
import os
import time
import argparse
import librosa

from unmix.source.data.song import Song
from unmix.source.configuration import Configuration
from unmix.source.engine import Engine
from unmix.source.logging.logger import Logger
from unmix.source.data.dataloader import DataLoader
from unmix.source.helpers.masker import mask


if __name__ == "__main__":
    global config

    parser = argparse.ArgumentParser(
        description="Executes a training session.")
    parser.add_argument('--configuration', default='../../../configurations/default-mask.jsonc',
                        type=str, help="Environment and training configuration.")
    parser.add_argument('--workingdir', default='../../../', type=str,
                        help="Working directory (default: current directory).")

    args = parser.parse_args()
    start = time.time()

    Configuration.initialize(args.configuration, args.workingdir)
    Logger.initialize()

    Logger.h1("unmix.io Neuronal Network Training Application")
    Logger.info("Environment: %s" % Configuration.get('environment.name'))
    Logger.info("Collection: %s" % Configuration.get('collection.folder'))
    Logger.info("Model: %s" % Configuration.get('training.model.name'))
    Logger.info("Arguments: ", str(args))

    engine = Engine()
    training_songs, validation_songs = DataLoader.load()
    song = Song(training_songs[0])
    vocals = song.vocals.load().channels
    instrumentals = song.instrumental.load().channels
    vocal_magnitude = np.abs(vocals)
    instrumental_magnitude = np.abs(instrumentals)
    target_mask = np.empty_like(vocal_magnitude)
    target_mask[instrumental_magnitude <= vocal_magnitude] = 1
    target_mask[instrumental_magnitude > vocal_magnitude] = 0

    mix_magnitude = vocal_magnitude + instrumental_magnitude
    target_mask_ratio = mask(vocal_magnitude, mix_magnitude)

    prediction_binary = mix_magnitude * target_mask
    prediction_ratio = mix_magnitude * target_mask_ratio

    mix_complex = vocals + instrumentals
    predicted_vocals_binary = prediction_binary * np.exp(np.angle(mix_complex) * 1j)
    predicted_vocals_ratio = prediction_ratio * np.exp(np.angle(mix_complex) * 1j)

    data_binary = librosa.istft(predicted_vocals_binary[0])
    audio_binary = np.array(data_binary)
    data_ratio = librosa.istft(predicted_vocals_ratio[0])
    audio_ratio = np.array(data_ratio)

    output_file_binary = os.path.join(Configuration.get_path('collection.folder'), os.path.basename(training_songs[0]) + "_predicted_binary.wav")
    Logger.info("Output prediction file: %s." % output_file_binary)
    librosa.output.write_wav(output_file_binary, audio_binary, 11025, norm=False)

    output_file_ratio = os.path.join(Configuration.get_path('collection.folder'),
                                      os.path.basename(training_songs[0]) + "_predicted_ratio.wav")
    Logger.info("Output prediction file: %s." % output_file_ratio)
    librosa.output.write_wav(output_file_ratio, audio_ratio, 11025, norm=False)
