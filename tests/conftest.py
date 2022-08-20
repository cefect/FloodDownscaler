'''
Created on Feb. 21, 2022

@author: cefect
'''
import os, shutil, random
import pytest
import numpy as np
import logging
from hp.oop import Session
from hp.logr import get_new_console_logger, logging
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
def session(tmp_path,out_dir, 
            logger,  
            test_name,
            
 
 
                    ):
    """Mock session for tests"""
 
    np.random.seed(100)
    random.seed(100)
    
 
    
    with Session( 
 
                 
                 #oop.Basic
                 out_dir=out_dir, 
                 tmp_dir=os.path.join(tmp_path, 'tmp_dir'),
                 
                  proj_name='testr', #probably a better way to propagate through this key 
                 run_name=test_name[:8].replace('_',''),
                  
                 relative=True, write=True, #avoid writing prep layers
                 
                 logger=logger, overwrite=True,
                   
                   #oop.Session
                   exit_summary=False,logfile_duplicate=False,
 
 
                   ) as ses:
        
        #ses.valid_dir=valid_dir
        assert len(ses.data_d)==0
        assert len(ses.compiled_fp_d)==0
        assert len(ses.ofp_d)==0
        yield ses
    

#===============================================================================
# @pytest.fixture(scope='session')
# def write():
#     write=False
#     if write:
#         print('WARNING!!! runnig in write mode')
#     return write
# 
# @pytest.fixture(scope='session')
# def logger():
#     out_dir = r'C:\LS\10_OUT\2112_Agg\outs\tests'
#     if not os.path.exists(out_dir): os.makedirs(out_dir)
#     os.chdir(out_dir) #set this to the working directory
#     print('working directory set to \"%s\''%os.getcwd())
# 
#     from hp.logr import BuildLogr
#     lwrkr = BuildLogr()
#     return lwrkr.logger
#===============================================================================
 
 
 