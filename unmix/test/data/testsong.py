#!/usr/bin/env python3
# coding: utf8

"""
Tests loading songs and tracks.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"


import numpy as np

from unmix.source.choppers.chopper import Chopper
import unmix.source.helpers.reducer as reducer
from unmix.source.data.song import Song


def test_load_song_mono():
    song = Song(
        "D:\\Data\\unmix.io\\4_training\\fft-window=1536_sample-rate=11025_channels=1-mono\\musdb18\\Steven Clark - Bounty")
    mix, vocals = song.load()
    assert len(vocals) > 0
    assert not song.vocals.initialized
    assert len(song.vocals.channels) == 0
    assert len(mix) > 0
    assert not song.mix.initialized
    assert len(song.mix.channels) == 0
    assert not np.array_equal(mix, vocals)


def test_load_chopper_horizontal_song_mono():
    song = Song(
        "D:\\Data\\unmix.io\\4_training\\fft-window=1536_sample-rate=11025_channels=1-mono\\musdb18\\Steven Clark - Bounty")
    chopper = Chopper(Chopper.DIRECTION_HORIZONTAL, Chopper.MODE_SPLIT, 64)
    mix, vocals = song.load(chopper)
    assert len(vocals) > 0
    assert not song.vocals.initialized
    assert len(song.vocals.channels) == 0
    assert len(mix) > 0
    assert not song.mix.initialized
    assert len(song.mix.channels) == 0
    assert not np.array_equal(mix, vocals)


if __name__ == "__main__":
    test_load_song_mono()
    test_load_chopper_horizontal_song_mono()
    print("Test run successful.")
