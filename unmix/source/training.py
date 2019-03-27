"""
Executes a training session.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"


import os
import time
import argparse

from unmix.source.helpers import console
from unmix.source.unmixnet import UnmixNet
from unmix.source.configuration import Configuration


if __name__ == "__main__":
    global config

    parser = argparse.ArgumentParser(description="Executes a training session.")
    parser.add_argument('--configuration', default='./configurations/default.json',
                        type=str, help="Environment and training configuration.")
    parser.add_argument('--workingdir', default=os.getcwd(), 
                        type=str, help="Working directory (default: current directory)")
    parser.add_argument('--working_mode', default='train', type=str,
                        help="Enter 'train' or 'separate', if entered 'separate', songfile in --songfile parameter must be provided")
    parser.add_argument('--songfiles', default='', type=str, help="Audiofiles to be splited")

    args = parser.parse_args()
    console.info("Arguments: ", str(args))

    if args.working_mode == 'train':
        start = time.time()

        config = Configuration.initialize(args.configuration, args.workingdir)

        console.h1("unmix.io Neuronal Network Training Application")
        console.info("Environment: %s" % Configuration.get('environment.name'))
        console.info("Collection: %s" % Configuration.get('collection.folder'))
        console.info("Model: %s" % Configuration.get('training.model.name'))

        unmixnet = UnmixNet()
        unmixnet.train()
        # Todo: unmixnet.save_weights()

        end = time.time()
        console.info("Finished processing in %d [ms]." % (end - start))
    elif args.working_mode == 'separate':
        start = time.time()

        config = Configuration.initialize(args.configuration, args.workingdir)

        console.h1("unmix.io Neuronal Network Training Application")
        unmixnet = UnmixNet()
        unmixnet.load_weights(Configuration.get_path(os.path.join(Configuration.get_path('environment.weights.folder'), config.file_name)))
        # ToDo Implement song processing
        end = time.time()
    else:
        console.info("For parameter --workingmode either 'train' or 'separate' must be entered")

