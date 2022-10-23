'''
Created on Oct. 23, 2022

@author: cefect
'''
import pytest, tempfile, datetime, os, copy, logging
from hp.Q import *
#===============================================================================
# fixtures
#===============================================================================
@pytest.fixture(scope='session')
def logger(): 
    logger= logging.getLogger('root')
    logger.setLevel(logging.DEBUG) #set the base level for the logger
    logger.info('built test session logger')
    
    return logger


@pytest.mark.dev
 
def test_session(logger):
    
    QSession(logger=logger)