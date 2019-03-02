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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Executes a training session.')
    parser.add_argument('--configuration', default='\\configurations\\default.json', type=str, help='Environment and training configuration.')
        
    args = parser.parse_args()
    print('Arguments:', str(args))

    
    start = time.time()
    
    end = time.time()
    
    print('Finished processing in %d [ms]', (end - start))
