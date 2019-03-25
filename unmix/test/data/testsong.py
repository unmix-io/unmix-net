"""
Tests loading songs and tracks.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"

import numpy as np

from unmix.source.data.song import Song
from unmix.source.choppers.chopper import Chopper

def test_load_song():
    song = Song("D:\\Data\\unmix.io\\4_training\\fft-window=1536_sample-rate=11025_channels=1-mono\\musdb18\\Steven Clark - Bounty")
    vocals = song.load_vocals()
    assert len(vocals) > 0
    assert song.vocals.initialized
    assert len(song.vocals.channels) > 0
    mix = song.load_mix()
    assert len(mix) > 0
    assert song.mix.initialized
    assert len(song.mix.channels) > 0
    assert song.vocals.channels[0].shape == song.mix.channels[0].shape

def test_load_chopper_horizontal_song():
    song = Song("D:\\Data\\unmix.io\\4_training\\fft-window=1536_sample-rate=11025_channels=1-mono\\musdb18\\Steven Clark - Bounty")
    choppers = [Chopper(Chopper.DIRECTION_HORIZONTAL, Chopper.MODE_SPLIT, 64)]
    
    vocals = song.load_vocals(choppers)
    assert len(vocals) > 0
    assert song.vocals.initialized
    assert len(song.vocals.channels) > 0
    mix = song.load_mix(choppers)
    assert len(mix) > 0
    assert song.mix.initialized
    assert len(song.mix.channels) > 0
    assert song.vocals.channels[0].shape == song.mix.channels[0].shape


if __name__ == "__main__":
    test_load_song()
    test_load_chopper_horizontal_song()
