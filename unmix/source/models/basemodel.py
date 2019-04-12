#!/usr/bin/env python3
# coding: utf8

"""
Base model which all models inherit from
"""

class BaseModel(object):
    @property
    def name(self):
        raise NotImplementedError
    
    def build(self):
        raise NotImplementedError