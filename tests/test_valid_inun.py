'''
Created on Mar. 3, 2023

@author: cefect
'''

#===============================================================================
# IMPORTS-----
#===============================================================================
import pytest, copy, os, random, re
xfail = pytest.mark.xfail

import numpy as np
import pandas as pd
import geopandas as gpd

from hp.tests.tools.rasters import get_poly_fp_from_rlay

from tests.conftest import (
      proj_lib, get_rlay_fp, crs_default
    )


from fdsc.analysis.valid.v_inun import ValidateMask
#===============================================================================
# test data-------
#===============================================================================
from tests.data.toy import wse1_arV, wse1_ar3
 

td1 = proj_lib['fred01']

wse1_rlay3_fp = get_rlay_fp(wse1_ar3, 'wse13')
wse1_rlayV_fp = get_rlay_fp(wse1_arV, 'wse1V')
dem1_rlay_fp = get_rlay_fp(wse1_arV, 'dem1')

inun_poly_fp = get_poly_fp_from_rlay(wse1_rlayV_fp)

#===============================================================================
# fixtures------------
#===============================================================================
@pytest.fixture(scope='function')
def wrkr(logger, tmp_path):
    with ValidateMask(logger=logger, 
                 ) as ses:
        yield ses

#===============================================================================
# tests.inundation ----------
#===============================================================================
@pytest.mark.parametrize('true_inun_fp, pred_inun_fp', [
    (td1['wse1_rlayV_fp'], td1['wse1_rlay3_fp']),
    (wse1_rlayV_fp, wse1_rlay3_fp),
    (inun_poly_fp, wse1_rlay3_fp),
    ]) 
def test_ValidateMask_init(true_inun_fp, pred_inun_fp,
                         logger):
    """just the validation worker init"""
    
    with ValidateMask(true_inun_fp, pred_inun_fp,  logger=logger) as wrkr:
        pass
    
    
toy_tp_mars = (~wse1_arV.mask, ~wse1_ar3.mask) #shortchutting
@pytest.mark.parametrize('true_mar, pred_mar',[
    toy_tp_mars,
    ])
def test_confusion_ser(true_mar, pred_mar, wrkr):
    wrkr._confusion(true_mar=true_mar, pred_mar=pred_mar)
    

@pytest.mark.parametrize('true_mar, pred_mar',[
    toy_tp_mars,
    ])
def test_hitRate(true_mar, pred_mar, wrkr):
    hitRate = wrkr.get_hitRate(true_mar=true_mar, pred_mar=pred_mar)
    
    #===========================================================================
    # #check
    #===========================================================================
    
    m1b1 = np.logical_and(pred_mar, true_mar)
    m0b1 = np.logical_and(np.invert(pred_mar), true_mar)
    
    assert hitRate == m1b1.sum() / (m1b1.sum() + m0b1.sum())
    
 
@pytest.mark.parametrize('true_mar, pred_mar',[
    toy_tp_mars,
    ])
def test_falseAlarm(true_mar, pred_mar, wrkr):
    falseAlarm = wrkr.get_falseAlarms(true_mar=true_mar, pred_mar=pred_mar)
    
    #===========================================================================
    # #check
    #===========================================================================
    
    true_arB, pred_arB =true_mar, pred_mar
    
    m1b0 = np.logical_and(pred_arB, np.invert(true_arB))
    m1b1 = np.logical_and(pred_arB, true_arB)
 
    
    assert falseAlarm == m1b0.sum()/(m1b1.sum()+m1b0.sum())
    

@pytest.mark.parametrize('true_mar, pred_mar',[
    toy_tp_mars,
    ])
def test_criticalSuccessIndex(true_mar, pred_mar, wrkr):
    csi = wrkr.get_criticalSuccessIndex(true_mar=true_mar, pred_mar=pred_mar)
    
    #===========================================================================
    # #check
    #===========================================================================
    
    true_arB, pred_arB =true_mar, pred_mar
    
    m1b0 = np.logical_and(pred_arB, np.invert(true_arB))
    m0b1 = np.logical_and(np.invert(pred_arB), true_arB)
    m1b1 = np.logical_and(pred_arB, true_arB)
 
    
    assert csi == m1b1.sum()/(m1b1.sum()+m1b0.sum()+m0b1.sum())

 
@pytest.mark.parametrize('true_mar, pred_mar',[
    toy_tp_mars,
    ])
def test_errorBias(true_mar, pred_mar, wrkr):
    errorBias = wrkr.get_errorBias(true_mar=true_mar, pred_mar=pred_mar)
    
    #===========================================================================
    # #check
    #===========================================================================
    
    true_arB, pred_arB =true_mar, pred_mar
    
    m1b0 = np.logical_and(pred_arB, np.invert(true_arB))
    m0b1 = np.logical_and(np.invert(pred_arB), true_arB)
 
    
    assert errorBias == m1b0.sum()/m0b1.sum()
    

@pytest.mark.parametrize('true_mar, pred_mar',[
    toy_tp_mars,
    ])
def test_inundation_all(true_mar, pred_mar, wrkr):
    res_d = wrkr.get_inundation_all(true_mar=true_mar, pred_mar=pred_mar)
    
    res_d.keys()
    
  
 
@pytest.mark.parametrize('true_inun_fp, pred_inun_fp', [
    #(td1['wse1_rlayV_fp'], td1['wse1_rlay3_fp']),
    (wse1_rlayV_fp, wse1_rlay3_fp),
    
    ]) 
def test_get_confusion_grid(true_inun_fp, pred_inun_fp, logger, tmp_path):
    """just the validation worker init"""
    
    with ValidateMask(true_inun_fp=true_inun_fp, pred_inun_fp=pred_inun_fp, 
                        logger=logger,out_dir=tmp_path,tmp_dir=os.path.join(tmp_path, 'tmp_dir'),
                        fancy_name='test',
                 ) as wrkr:
        conf_ar = wrkr.get_confusion_grid()
        wrkr.write_array(conf_ar)


    
    
    