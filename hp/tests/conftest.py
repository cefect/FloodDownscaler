'''
Created on Feb. 21, 2022

@author: cefect

#USE-------------
#import all the coms fixtures
from hp.tests.conftest import *

Pytest locates conftest.py files by searching for them in the test root path and all 
of its parent directories 1.
 The conftest.py file serves as a means of providing fixtures for an entire directory. 
 Fixtures defined in a conftest.py can be used by any test in that package without needing to import them 
 (pytest will automatically discover them)

'''
import os, shutil, random, datetime, tempfile
from pyproj.crs import CRS
import pytest
import numpy as np
import logging
from hp.oop import Session
from hp.logr import get_new_console_logger, logging

#===============================================================================
# vars
#===============================================================================
temp_dir = os.path.join(tempfile.gettempdir(), __name__, datetime.datetime.now().strftime('%Y%m%d'))
if not os.path.exists(temp_dir): os.makedirs(temp_dir)


#===============================================================================
# fixture-----
#===============================================================================

    
@pytest.fixture(scope='session')
def logger():
    return get_new_console_logger(level=logging.DEBUG)
    
    
@pytest.fixture(scope='function')
def out_dir(tmp_path):
    return tmp_path

@pytest.fixture(scope='function')
def test_name(request):
    return request.node.name.replace('[','_').replace(']', '_')


@pytest.fixture(scope='function')
def init_kwargs(tmp_path,logger, test_name):
    return dict(
        out_dir=tmp_path, 
        tmp_dir=os.path.join(tmp_path, 'tmp_dir'),
        base_dir=tmp_path,
        #prec=prec,
        proj_name='test', #probably a better way to propagate through this key 
        run_name=test_name[:8].replace('_',''),
        
        relative=True, 
        
        logger=logger, overwrite=True, logfile_duplicate=False,
        )