'''
Created on Feb. 21, 2022

@author: cefect
'''
import os, shutil
import pytest
import numpy as np
 
 
    
#===============================================================================
# fixture-----
#===============================================================================
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
 
 
 