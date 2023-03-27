'''
Created on Mar. 27, 2023

@author: cefect
'''

import pytest, copy, os, random, re
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