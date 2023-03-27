'''
Created on Mar. 27, 2023

@author: cefect
'''

import pytest, copy, os, random, re, pickle
import numpy as np
import pandas as pd

xfail = pytest.mark.xfail

from fdsc.eval.control import Dsc_Eval_Session as Session


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

@pytest.mark.parametrize('pick_fp',)
                         
def test_run_dsc_vali_multi(pick_fp, ses):
    
    with open(pick_fp, "rb") as f:
        res_lib = pickle.load(f)
    
    ses.run_dsc_vali_multi(res_lib)