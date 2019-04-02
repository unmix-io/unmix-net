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

from unmix.source.helpers import console
from unmix.source.engine import Engine
from unmix.source.configuration import Configuration


if __name__ == "__main__":
    global config

    parser = argparse.ArgumentParser(
        description="Executes a training session.")
    parser.add_argument('--configuration', default='./configurations/default.jsonc',
                        type=str, help="Environment and training configuration.")
    parser.add_argument('--workingdir', default=os.getcwd(),
                        type=str, help="Working directory (default: current directory).")

    args = parser.parse_args()
    console.info("Arguments: ", str(args))
    start = time.time()

    config = Configuration.initialize(args.configuration, args.workingdir)

    console.h1("unmix.io Neuronal Network Training Application")
    console.info("Environment: %s" % Configuration.get('environment.name'))
    console.info("Collection: %s" % Configuration.get('collection.folder'))
    console.info("Model: %s" % Configuration.get('training.model.name'))

    engine = Engine()

    if Configuration.get('training.load_weights'):
        engine.load_weights()

    engine.train()

    end = time.time()
    console.info("Finished processing in %d [s]." % (end - start))
