"""
Executes a training session.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"

import os
import sys
import time
import signal
import random
import string
import argparse
import numpy as np

from helpers import console
from configuration import Configuration
from models.modelfactory import ModelFactory
from optimizers.optimizerfactory import OptimizerFactory
from lossfunctions.lossfunctionfactory import LossFunctionFactory

if __name__ == "__main__":
    global config

    parser = argparse.ArgumentParser(description='Executes a training session.')
    parser.add_argument('--configuration', default='D:\\Repos\\unmix.io\\unmix-net\\configurations\\default.json', type=str, help='Environment and training configuration.')

    args = parser.parse_args()
    console.info('Arguments: ', str(args))

    start = time.time()

    config = Configuration.initialize(args.configuration)
    model = ModelFactory.build()
    optimizer = OptimizerFactory.build()
    loss_function = LossFunctionFactory.build()
    model.compile(loss=loss_function, optimizer=optimizer, metrics=metrics)
    model.summary(line_length=150)

    console.debug('Model initialized with %d parameters' % model.count_params())

    end = time.time()

    console.info('Finished processing in %d [ms]' % (end - start))
