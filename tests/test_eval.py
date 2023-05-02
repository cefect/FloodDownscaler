'''
Created on Mar. 27, 2023

@author: cefect
'''

import pytest, copy, os, random, re, pickle
import numpy as np
import pandas as pd

xfail = pytest.mark.xfail

from fdsc.eval.control import Dsc_Eval_Session as Session
#from definitions import src_dir
"""issues with relative paths"""
src_dir = r'C:\LS\09_REPOS\\03_TOOLS\\FloodDownscaler'
#===============================================================================
# test data
#===============================================================================
#downsample results
dsc_pick_pars = ('pick_fp_rel',[(r'tests\data\test_run_dsc_multi_method_pars_toy\test_testrun_0327_dscM.pkl')])


from fperf.tests.test_pipe import toy_hwm_fp, toy_d, toy_aoi_fp
                         
toy_dsc_d = {
    #output from _get_fps_from_dsc_lib
    'dsc_pick_fp':os.path.join(src_dir, r'tests\data\test_run_dsc_multi_method_pars_toy\compiled.pkl'),
    
    #toy observation data from fperf.tests
    'aoi':toy_aoi_fp,
    'hwm':toy_hwm_fp,
    'inun':toy_d['inunP']    
    }
#===============================================================================
# helpers
#===============================================================================
def _get_rel_fps(pick_fp_rel):
    pick_fp = os.path.join(src_dir, pick_fp_rel)
    with open(pick_fp, "rb") as f:
        dsc_res_lib = pickle.load(f)
    base_dir = os.path.dirname(pick_fp)
    return dsc_res_lib, base_dir

#===============================================================================
# fixtures------------
#===============================================================================
@pytest.fixture(scope='function')
def ses(init_kwargs):
    with Session(**init_kwargs) as session:
        yield session
        
        
#===============================================================================
# tetss------
#===============================================================================
def test_init(ses):
    pass

 
@pytest.mark.parametrize(*dsc_pick_pars)                         
def test_get_fps_from_dsc_lib(pick_fp_rel, ses):
    dsc_res_lib, base_dir = _get_rel_fps(pick_fp_rel)    
    
    res_d = ses._get_fps_from_dsc_lib(dsc_res_lib, base_dir=base_dir)
    
    #===========================================================================
    # with open(r'l:\09_REPOS\03_TOOLS\FloodDownscaler\tests\data\test_run_dsc_multi_method_pars_toy\compiled.pkl',
    #           'wb') as file:
    #     pickle.dump(res_d, file)
    #===========================================================================
    
    
@pytest.mark.dev
@pytest.mark.parametrize('pick_fp, hwm_pts_fp, inun_fp, aoi_fp', [
    [toy_dsc_d[k] for k in ['dsc_pick_fp', 'hwm', 'inun', 'aoi']
    ]])                      
def test_run_dsc_vali_multi(pick_fp, hwm_pts_fp, inun_fp, aoi_fp, ses):
    #load the results filepath pickle
    with open(pick_fp, "rb") as f:
        fp_lib = pickle.load(f)  
    
    ses.run_vali_multi_dsc(fp_lib, 
 
                           hwm_pts_fp=hwm_pts_fp, inun_fp=inun_fp
                           )
    
    
    
    
    