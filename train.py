#!/usr/bin/env python3
# coding: utf8

"""
               .---._
           .--(. '  .).--.      . .-.
        . ( ' _) .)` (   .)-. ( ) '-'
       ( ,  ).        `(' . _)
     (')  _________      '-'
     ____[_________]                                         __________
     \__/ | _ \  ||    ,;,;,,                               [__________]
     _][__|(")/__||  ,;;;;;;;;,   __________   __________   _| unmix.io |_
    /             | |____      | |          | |  ___     | |        ____|
   (| .--.    .--.| |     ___  | |   |  |   | |      ____| |____        |
   /|/ .. \~~/ .. \_|_.-.__.-._|_|_.-:__:-._|_|_.-.__.-._|_|_.-.__.-.___|
+=/_|\ '' /~~\ '' /=+( o )( o )+==( o )( o )=+=( o )( o )+==( o )( o )=+=
='=='='--'==+='--'===+'-'=='-'==+=='-'+='-'===+='-'=='-'==+=='-'=+'-'jgs+

Executes a training session.
"""


__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"


import os
import time
import argparse

from unmix.source.configuration import Configuration
from unmix.source.engine import Engine
from unmix.source.logging.logger import Logger


if __name__ == "__main__":
    global config

    parser = argparse.ArgumentParser(
        description="Executes a training session.")
    parser.add_argument('--configuration', default='',
                        type=str, help="Environment and training configuration.")
    parser.add_argument('--workingdir', default=os.getcwd(), type=str, help="Working directory (default: current directory).")
    parser.add_argument('--model', default='', type=str, help="Optional pretrained model to continue training.")
    parser.add_argument('--weights', default='', type=str, help="Optional pretrained weights of a model to continue training.")


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

    if args.model:
        engine.load(args.model)
    if args.weights:
        engine.load_weights(args.weights)

    engine.plot_model()
    engine.train()

    end = time.time()
    Logger.info("Finished processing in %d [s]." % (end - start))
