'''
Created on Mar. 27, 2023

@author: cefect
'''

import pytest, copy, os, random, re, pickle
import numpy as np
import pandas as pd

xfail = pytest.mark.xfail

from fdsc.eval.control import Dsc_Eval_Session as Session
from definitions import src_dir

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

@pytest.mark.dev
@pytest.mark.parametrize('pick_fp_rel',
                         [(r'tests\data\test_run_dsc_multi_method_pars_toy\test_testrun_0327_dscM.pkl')],
                         )
                         
def test_get_fps_from_dsc_lib(pick_fp_rel, ses):
    pick_fp = os.path.join(src_dir, pick_fp_rel)
    with open(pick_fp, "rb") as f:
        dsc_res_lib = pickle.load(f)
        
    
    
    ses._get_fps_from_dsc_lib(dsc_res_lib, base_dir=os.path.dirname(pick_fp))