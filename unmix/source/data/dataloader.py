#!/usr/bin/env python3
# coding: utf8

"""
Loads and handels training and validation data collections.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"


import hashlib
import glob
import os
import random

from unmix.source.configuration import Configuration
from unmix.source.data.song import Song
from unmix.source.logging.logger import Logger


class DataLoader(object):

    TEST_DATA_COUNT = 50

    @staticmethod
    def load():
        path = Configuration.get_path('collection.folder', False)
        files = [os.path.dirname(file) for file in glob.iglob(
            os.path.join(path, '**', '%s*.h5' % Song.PREFIX_VOCALS), recursive=True)]

        # Sort files by hash value of folder to guarantee a consistent order
        files.sort(key=lambda x: hashlib.md5(
            os.path.basename(x).encode('utf-8')).hexdigest())

        test_files = None
        test_frequency = Configuration.get('collection.test_frequency')
        if test_frequency and test_frequency > 0:
            test_files = files[-DataLoader.TEST_DATA_COUNT:]
            files = files[:len(files) - DataLoader.TEST_DATA_COUNT]

        songs_limit = Configuration.get('collection.songs_limit')

        if songs_limit and songs_limit > 0:
            if songs_limit <= 1:  # Configuration as percentage share
                songs_limit = songs_limit * len(files)
            songs_limit = min(songs_limit, len(files))
            files = files[:songs_limit]

        validation_ratio = Configuration.get('collection.validation_ratio')
        validation_files = files[:int(len(files) * validation_ratio)]
        training_files = files[len(validation_files):]

        Logger.debug("Found %d songs for traing and %d songs for validation."
                     % (len(training_files), len(validation_files)))
        if test_files:
            Logger.debug("Use %d songs for tests after every %d epoch."
                         % (len(test_files), test_frequency))

        return training_files, validation_files, test_files
