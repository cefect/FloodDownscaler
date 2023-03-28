'''
Created on Mar. 27, 2023

@author: cefect
'''


import pytest, copy, os, random, re, pickle
import numpy as np
import pandas as pd

xfail = pytest.mark.xfail

from definitions import src_dir
if 'L:' in src_dir:
    """issues with relative paths"""
    src_dir = src_dir.replace('L:', 'C:\\LS')
    
from hp.pd import view
from hp.tests.tools.rasters import get_rlay_fp

from fdsc.plot.control import Fdsc_Plot_Session as Session

#tests from fperf
 

#===============================================================================
# test data
#===============================================================================
from fperf.tests.data.toy import dem1_ar, crs_default, bbox_default
#===============================================================================
# gfp = lambda ar, name:get_rlay_fp(ar, name, crs=crs_default, bbox=bbox_default)
# toy_d = {'dem':gfp(dem1_ar, 'dem1')}
#===============================================================================

#see tests.test_eval.test_run_dsc_vali_multi()
vali_multi_pick_fp = os.path.join(src_dir, 
          r'tests\data\test_run_dsc_vali_multi_toy\test_testrun_0328_gfps.pkl')

#===============================================================================
# fixtures------------
#===============================================================================

@pytest.fixture(scope='function')
def ses(init_kwargs):    
    """Mock session for tests""" 
    with Session(**init_kwargs, crs=crs_default) as session: 
        yield session
        
#===============================================================================
# TESTS---------
#===============================================================================
def test_init(ses):
    pass

@pytest.mark.dev
@pytest.mark.parametrize('pick_fp', [vali_multi_pick_fp])
def test_load_run_serx(pick_fp, ses):
    """same as fperf.tests.test_plot.test_load_run_serx()"""
     
    assert os.path.exists(pick_fp)
    serx = ses.load_run_serx(pick_fp, base_dir=os.path.dirname(pick_fp))
    
    """
    view(serx)
    """
    
    serx.to_pickle(
        os.path.join(os.path.dirname(vali_multi_pick_fp), 'serx.pkl') 
        )
