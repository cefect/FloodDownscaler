'''
Created on May 18, 2019

@author: cef

custom exceptions and errors
'''

import logging
mod_logger = logging.getLogger('exceptions') #creates a child logger of the root

import warnings



class Error(Exception):
    """Base class for exceptions in this module."""
    def __init__(self, msg):
        mod_logger.error(msg)

def assert_func(func, msg=''):
    if __debug__: # true if Python was not started with an -O option
        result, msgf = func()
        assert result, msg+': '+msgf