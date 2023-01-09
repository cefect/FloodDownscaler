'''
Created on Jan. 6, 2023

@author: cefect
'''

import pytest, copy, os, random, re
xfail = pytest.mark.xfail

import numpy as np


from fdsc.scripts.valid import ValidateWorker, ValidateSession, run_validator

from tests.conftest import (
      proj_lib, get_rlay_fp 
    )

#===============================================================================
# test data-------
#===============================================================================
from tests.data.toy import wse1_arV, wse1_ar3

wse1_rlay3_fp = get_rlay_fp(wse1_ar3, 'wse13')
wse1_rlayV_fp = get_rlay_fp(wse1_arV, 'wse1V')
#===============================================================================
# fixtures------------
#===============================================================================
@pytest.fixture(scope='function')
def wrkr(logger, tmp_path):
    with ValidateWorker(logger=logger,
                                         #oop.Basic

                 ) as ses:
        yield ses
    
#===============================================================================
# tests--------
#===============================================================================


@pytest.mark.parametrize('true_fp, pred_fp', [
    #(proj_lib['fred01']['wse1_rlayV_fp'], proj_lib['fred01']['wse1_rlay3_fp']),
    (wse1_rlayV_fp, wse1_rlay3_fp),
    ]) 
def test_valid_wrkr_init(true_fp, pred_fp, logger):
    """just the validation worker init"""
    
    with ValidateWorker(true_fp, pred_fp, logger=logger) as wrkr:
        pass
    
    
 
@pytest.mark.parametrize('true_ar, pred_ar',[
    (wse1_arV, wse1_ar3),
    ])
def test_hitRate(true_ar, pred_ar, wrkr):
    hitRate = wrkr.get_hitRate(true_ar=true_ar, pred_ar=pred_ar)
    
    #===========================================================================
    # #check
    #===========================================================================
    
    true_arB, pred_arB = np.invert(true_ar.mask), np.invert(pred_ar.mask)
    
    m1b1 = np.logical_and(pred_arB, true_arB)
    m0b1 = np.logical_and(np.invert(pred_arB), true_arB)
    
    assert hitRate == m1b1.sum()/(m1b1.sum()+m0b1.sum())
        
    
 
@pytest.mark.parametrize('true_ar, pred_ar',[
    (wse1_arV, wse1_ar3),
    ])
def test_falseAlarm(true_ar, pred_ar, wrkr):
    falseAlarm = wrkr.get_falseAlarms(true_ar=true_ar, pred_ar=pred_ar)
    
    #===========================================================================
    # #check
    #===========================================================================
    
    true_arB, pred_arB = np.invert(true_ar.mask), np.invert(pred_ar.mask)
    
    m1b0 = np.logical_and(pred_arB, np.invert(true_arB))
    m1b1 = np.logical_and(pred_arB, true_arB)
 
    
    assert falseAlarm == m1b0.sum()/(m1b1.sum()+m1b0.sum())
    

@pytest.mark.parametrize('true_ar, pred_ar',[
    (wse1_arV, wse1_ar3),
    ])
def test_criticalSuccessIndex(true_ar, pred_ar, wrkr):
    csi = wrkr.get_criticalSuccessIndex(true_ar=true_ar, pred_ar=pred_ar)
    
    #===========================================================================
    # #check
    #===========================================================================
    
    true_arB, pred_arB = np.invert(true_ar.mask), np.invert(pred_ar.mask)
    
    m1b0 = np.logical_and(pred_arB, np.invert(true_arB))
    m0b1 = np.logical_and(np.invert(pred_arB), true_arB)
    m1b1 = np.logical_and(pred_arB, true_arB)
 
    
    assert csi == m1b1.sum()/(m1b1.sum()+m1b0.sum()+m0b1.sum())

 
@pytest.mark.parametrize('true_ar, pred_ar',[
    (wse1_arV, wse1_ar3),
    ])
def test_errorBias(true_ar, pred_ar, wrkr):
    errorBias = wrkr.get_errorBias(true_ar=true_ar, pred_ar=pred_ar)
    
    #===========================================================================
    # #check
    #===========================================================================
    
    true_arB, pred_arB = np.invert(true_ar.mask), np.invert(pred_ar.mask)
    
    m1b0 = np.logical_and(pred_arB, np.invert(true_arB))
    m0b1 = np.logical_and(np.invert(pred_arB), true_arB)
 
    
    assert errorBias == m1b0.sum()/m0b1.sum()
    

@pytest.mark.parametrize('true_ar, pred_ar',[
    (wse1_arV, wse1_ar3),
    ])
def test_inundation_all(true_ar, pred_ar, wrkr):
    d = wrkr.get_inundation_all(true_ar=true_ar, pred_ar=pred_ar)
    
  
 
@pytest.mark.parametrize('true_fp, pred_fp', [
    #(proj_lib['fred01']['wse1_rlayV_fp'], proj_lib['fred01']['wse1_rlay3_fp']),
    (wse1_rlayV_fp, wse1_rlay3_fp),
    ]) 
def test_get_confusion_grid(true_fp, pred_fp, logger, tmp_path):
    """just the validation worker init"""
    
    with ValidateSession(true_fp=true_fp, pred_fp=pred_fp, 
                        logger=logger,out_dir=tmp_path,tmp_dir=os.path.join(tmp_path, 'tmp_dir'),
                        fancy_name='test',
                 ) as wrkr:
        conf_ar = wrkr.get_confusion_grid()
        wrkr.write_array(conf_ar)

@pytest.mark.dev  
@pytest.mark.parametrize('true_fp, pred_fp', [
    #(proj_lib['fred01']['wse1_rlayV_fp'], proj_lib['fred01']['wse1_rlay3_fp']),
    (wse1_rlayV_fp, wse1_rlay3_fp),
    ])   
def test_run_validator(true_fp, pred_fp):
    run_validator(true_fp, pred_fp)
    
    
    