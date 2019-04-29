#!/usr/bin/env python3
# coding: utf8

"""
Normalizes data to zero mean and unit variance.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"


import json
import numpy as np
import math

from unmix.source.configuration import Configuration
from unmix.source.exceptions.configurationerror import ConfigurationError
from unmix.source.helpers import filehelper


name = 'normalizer_zmuv'

MODE_BIN = 'frequency_bin'
MODE_SINGLE = 'single'


def normalize(data, mode, config_file):
    'Normalizes training data to zero mean and unit variance.'
    config = __read_config(config_file)
    if mode == MODE_SINGLE:
        mean = np.array(config['mean'])
        variance = np.array(config['variance'])
    elif mode == MODE_BIN:
        mean = np.repeat(np.reshape(config['bin_mean'], (data.shape[0], 1, data.shape[2])), data.shape[1], axis=1)
        variance = np.repeat(np.reshape(config['bin_variance'], (data.shape[0], 1, data.shape[2])), data.shape[1], axis=1)
    else:        
        raise ConfigurationError('Invalid zmuv normalizer configuration.')
    
    return (data - mean) / np.sqrt(variance)


def denormalize(data,  mode, config_file):
    'Returns denormalized values aways with previous mean and variance.'
    config = __read_config(config_file)
    if mode == MODE_SINGLE:
        mean = np.array(config['mean'])
        variance = np.array(config['variance'])
    elif mode == MODE_BIN:
        mean = np.repeat(np.reshape(config['bin_mean'], (data.shape[0], 1, data.shape[2])), data.shape[1], axis=1)
        variance = np.repeat(np.reshape(config['bin_variance'], (data.shape[0], 1, data.shape[2])), data.shape[1], axis=1)
    else:        
        raise ConfigurationError('Invalid zmuv normalizer configuration.')

    return (data * np.sqrt(variance)) + mean


def __read_config(config_file):
    with open(filehelper.build_abspath(config_file, Configuration.get_path('collection.folder'))) as file:
        return json.load(file)
