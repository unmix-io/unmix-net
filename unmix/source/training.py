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

    args = parser.parse_args()
    console.info("Arguments: ", str(args))

    start = time.time()

    config = Configuration.initialize(args.configuration, args.workingdir)

    console.h1("unmix.io Neuronal Network Training Application")
    console.info("Environment: %s" % Configuration.get('environment.name'))
    console.info("Collection: %s" % Configuration.get('collection.folder'))
    console.info("Model: %s" % Configuration.get('training.model.name'))
    
    unmixnet = UnmixNet()
    unmixnet.train()
    end = time.time()

    console.info("Finished processing in %d [ms]." % (end - start))
