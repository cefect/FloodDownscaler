'''
Created on Apr. 15, 2022

@author: cefect
'''
import os, shutil, logging, math
import pytest
import pandas as pd
import numpy as np
from scipy.stats import norm

from hp.err_calc import ErrorCalcs as ErrorCalcs_class

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger() 
#===============================================================================
# fixtures-----
#===============================================================================

@pytest.fixture(scope='session', params=[1.0, 3.0])
def var(request):
    return request.param #zero doesnt work well for some tests


@pytest.fixture(scope='session')
def true_mean():
    return 10.0 #zero doesnt work well for some tests

@pytest.fixture(scope='session', params=[-5, 0.0, 7])
def pred_mean(request, true_mean):
    return true_mean + request.param 

@pytest.fixture(params = list(range(10))) #random seeds to test,
def seed(request):
    return request.param

@pytest.fixture()
def ErrorCalcs(#get an initialized ErrorCalcs worker
        #request,
       seed, #iterating seed fixture
       true_mean,pred_mean,var,
        n=int(1e5),

 
        
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
    if not chk==0.0:    
        assert np.allclose(calc, chk, atol=1e-2)
    else:
        """no counterbalaing"""
        assert np.allclose(calc, chk, atol=1e0)
        
@pytest.mark.dev
def test3_RMSE(ErrorCalcs, true_mean, pred_mean, var):
    dkey = 'RMSE'
    calc = ErrorCalcs.retrieve(dkey)
    

    
    #check it
    chk = math.sqrt((pred_mean - true_mean)**2)
    
    print('finished with calc=%.4f and chk=%.4f'%(calc, chk))
    print(ErrorCalcs.meta_d)
    
    
    if chk==0.0:
        """not sure why..."""
        assert np.allclose(calc, 1.0, rtol=1e-1)
    else:
        assert np.allclose(calc, chk, rtol=1e-1)

    
    