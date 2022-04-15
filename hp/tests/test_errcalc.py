'''
Created on Apr. 15, 2022

@author: cefect
'''
import os, shutil, logging
import pytest
import pandas as pd
import numpy as np
from scipy.stats import norm

from hp.errors import ErrorCalcs as ErrorCalcs_class

logger = logging.getLogger() 
#===============================================================================
# fixtures-----
#===============================================================================
@pytest.fixture(scope='session')
def true_mean():
    return 10.0 #zero doesnt work well for some tests

@pytest.fixture(scope='session', params=[5.0, 10.0, 15.0])
def pred_mean(request):
    return request.param 

@pytest.fixture(params = list(range(3))) #random seeds to test,
def seed(request):
    return request.param

@pytest.fixture()
def ErrorCalcs(#get an initialized ErrorCalcs worker
        #request,
       seed, #iterating seed fixture
       true_mean,pred_mean,
        n=int(1e5),
        var = 1.0,
 
        
           ): #build an ErrorCalcs
    
    #setup
    np.random.seed(seed)
    #pred_mean = request.param #synthetic prediction
    
    # build trues
    true_ser=pd.Series(true_mean, index=range(n))
    
    #build synthetic predictions 
    if var>0.0:
        #build noise
        rv_norm = norm(loc=pred_mean, scale=1.0)
        pred_ser = pd.Series(rv_norm.rvs(size=n), index=true_ser.index)
        
    elif var==0.0:
        pred_ser = pd.Series(pred_mean, index=true_ser.index)
    else:
        raise IOError(var)
 
    assert abs(pred_ser.mean() - pred_mean)<.01
 
    #build worker
    wrkr =  ErrorCalcs_class(pred_ser=pred_ser, 
                            true_ser=true_ser, 
                            logger=logger)
    
    #set info for debugging
    wrkr.meta_d = {'var':var, 'pred_mean':pred_mean, 'true_mean':true_mean, 'seed':seed}
 
    
    return wrkr
    

 

#===============================================================================
# tests---------
#==========R=====================================================================

def test0_bias(ErrorCalcs, true_mean, pred_mean):
    #do the calc
    dkey='bias'
    calc = ErrorCalcs.retrieve(dkey)
 
    #check it   
    chk = pred_mean/true_mean    
    assert np.allclose(calc, chk, atol=1e-2)
    
 
def test1_meanError(ErrorCalcs, true_mean, pred_mean):
    dkey = 'meanError'
    calc = ErrorCalcs.retrieve(dkey)
    
    #check it
    
    chk = pred_mean - true_mean
    assert np.allclose(calc, chk, atol=1e-2)
    
 
def test2_meanErrorAbs(ErrorCalcs, true_mean, pred_mean):
    dkey = 'meanErrorAbs'
    calc = ErrorCalcs.retrieve(dkey)
    
    #check it
    chk = abs(pred_mean - true_mean)    
    assert np.allclose(calc, chk, atol=1e-2)
    
    
    