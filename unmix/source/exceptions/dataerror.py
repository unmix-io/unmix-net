"""
Thrown if training data file is invalid.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"


class DataError(Exception):

    def __init__(self, file='', message=''):
        if file and message:
            super().__init__("Error for training file '%s': %s." % (file, message))
        elif file:
            super().__init__("Error for invalid or missing training file: %s." % file)
        else:
            super().__init__("Error while processing training data.")
