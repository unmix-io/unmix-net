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

from unmix.source.helpers import console
from unmix.source.unmixnet import UnmixNet
from unmix.source.configuration import Configuration


if __name__ == "__main__":
    global config

    parser = argparse.ArgumentPadescription="Executes a training session.")
    parser.add_argument('--configuration', default='D:\\Repos\\unmix.io\\unmix-net\\configurations\\default.json',
                        type=str, help="Environment and training configuration.")
    parser.add_argument('--workingdir', default=os.getcwd(), 
                        type=str, help="Working directory (default: current directory)")

    args = parser.parse_args()
    console.info("Arguments: ", str(args))

    start = time.time()

    config = Configuration.initialize(args.configuration, args.workingdir)

    unmixnet = UnmixNet()
    unmixnet.train(8)
    end = time.time()

    console.info("Finished processing in %d [ms]." % (end - start))
