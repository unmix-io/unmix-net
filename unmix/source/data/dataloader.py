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

    @staticmethod
    def load(path='', test_data_count=None):
        if not path:
            path = Configuration.get_path('collection.folder', False)
        files_vocal = [os.path.dirname(file) for file in glob.iglob(
            os.path.join(path, '**', '%s*.h5' % Song.PREFIX_VOCALS), recursive=True)]
        files_instrumental = [os.path.dirname(file) for file in glob.iglob(
            os.path.join(path, '**', '%s*.h5' % Song.PREFIX_INSTRUMENTAL), recursive=True)]
        
        files = [f for f in files_vocal if f in files_instrumental] # make sure both vocal and instrumental file exists
        skipped_count = len(set(files_vocal) - set(files_instrumental)) + len(set(files_instrumental) - set(files_vocal))
        Logger.debug(f"Skipped {skipped_count} files (incomplete vocal/instrumental pair)")

        # Sort files by hash value of folder to guarantee a consistent order
        files.sort(key=lambda x: hashlib.md5(
            os.path.basename(x).encode('utf-8', 'surrogatepass')).hexdigest())

        test_files = None
        test_frequency = Configuration.get('collection.test_frequency', default=0)
        if not test_data_count:
            test_data_count = Configuration.get('collection.test_data_count', default=0)
        if test_data_count > 0:
            test_data_count = int(test_data_count)
            test_files = files[-test_data_count:]
            files = files[:len(files) - test_data_count]

        song_limit = Configuration.get('collection.song_limit', default=0)
        if song_limit > 0:
            if song_limit <= 1:  # Configuration as percentage share
                song_limit = song_limit * len(files)
            song_limit = min(int(song_limit), len(files))
            files = files[:song_limit]

        validation_ratio = Configuration.get('collection.validation_ratio', default=0.2)
        validation_files = files[:int(len(files) * validation_ratio)]
        training_files = files[len(validation_files):]

        Logger.debug("Found %d songs for training and %d songs for validation."
                     % (len(training_files), len(validation_files)))
        if test_files:
            Logger.debug("Use %d songs for tests after every %d epoch."
                         % (len(test_files), test_frequency))

        if len(training_files) == 0:
            Logger.warn("No training files assigned.")
        if len(validation_files) == 0:
            Logger.warn("No validation files assigned.")
        return training_files, validation_files, test_files
