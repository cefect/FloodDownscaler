'''
Created on Mar. 27, 2023

@author: cefect
'''


import pytest, copy, os, random, re, pickle
import numpy as np
import pandas as pd

xfail = pytest.mark.xfail


#===============================================================================
# fixtures------------
#===============================================================================

@pytest.fixture(scope='function')
def ses(init_kwargs):    
    """Mock session for tests""" 
    with Session(**init_kwargs, crs=crs_default) as session: 
        yield session