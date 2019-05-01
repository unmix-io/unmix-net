#!/usr/bin/env python3
# coding: utf8

"""
Dictinoary helpers
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"

from collections import namedtuple, OrderedDict

# Deep merges dictionaries, by modifying destination in place. Source: https://stackoverflow.com/a/20666342/496950
def merge(source, destination):
    """
    run me with nosetests --with-doctest file.py

    >>> a = { 'first' : { 'all_rows' : { 'pass' : 'dog', 'number' : '1' } } }
    >>> b = { 'first' : { 'all_rows' : { 'fail' : 'cat', 'number' : '5' } } }
    >>> merge(b, a) == { 'first' : { 'all_rows' : { 'pass' : 'dog', 'fail' : 'cat', 'number' : '5' } } }
    True
    """
    for key, value in source.items():
        if isinstance(value, dict):
            # get node or create one
            node = destination.setdefault(key, {})
            merge(value, node)
        else:
            destination[key] = value

    return destination

def to_named_tuple(dictionary):
    if not isinstance(dictionary, dict):
        return dictionary
    return namedtuple("Config", dictionary.keys())(*[to_named_tuple(v) for v in dictionary.values()])
