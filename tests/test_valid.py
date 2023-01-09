'''
Created on Jan. 6, 2023

@author: cefect
'''

import pytest, copy, os, random, re
xfail = pytest.mark.xfail

import numpy as np
import pandas as pd
import geopandas as gpd


from fdsc.analysis.valid import ValidateGrid, ValidateSession, run_validator

from tests.conftest import (
      proj_lib, get_rlay_fp, crs_default
    )

#===============================================================================
# test data-------
#===============================================================================
from tests.data.toy import wse1_arV, wse1_ar3
td1 = proj_lib['fred01']

wse1_rlay3_fp = get_rlay_fp(wse1_ar3, 'wse13')
wse1_rlayV_fp = get_rlay_fp(wse1_arV, 'wse1V')
#===============================================================================
# fixtures------------
#===============================================================================
@pytest.fixture(scope='function')
def wrkr(logger, tmp_path):
    with ValidateGrid(logger=logger,
                                         #oop.Basic

                 ) as ses:
        yield ses
        

@pytest.fixture(scope='function')
def ses(tmp_path,write,logger, test_name,
         crs= crs_default,
                    ):
    
    """Mock session for tests"""
 
    #np.random.seed(100)
    #random.seed(100)
    
    with ValidateSession(
                 #oop.Basic
                 out_dir=tmp_path, 
                 tmp_dir=os.path.join(tmp_path, 'tmp_dir'),
                  proj_name='test', #probably a better way to propagate through this key 
                 run_name=test_name[:8].replace('_',''),                  
                 relative=True, write=write, #avoid writing prep layers                 
                 logger=logger, overwrite=True,
                   
                   #oop.Session
                   logfile_duplicate=False,
                   
                   #RioSession
                   crs=crs, 
                   ) as ses:
 
        yield ses
    
#===============================================================================
# tests--------
#===============================================================================


@pytest.mark.parametrize('true_fp, pred_fp', [
    (td1['wse1_rlayV_fp'], td1['wse1_rlay3_fp']),
    (wse1_rlayV_fp, wse1_rlay3_fp),
    ]) 
def test_valid_wrkr_init(true_fp, pred_fp,
                         logger):
    """just the validation worker init"""
    
    with ValidateGrid(true_fp, pred_fp,  logger=logger) as wrkr:
        pass
    
    
#===============================================================================
# tests.inundation ----------
#===============================================================================
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
    res_d = wrkr.get_inundation_all(true_ar=true_ar, pred_ar=pred_ar)
    
    res_d.keys()
    
  
 
@pytest.mark.parametrize('true_fp, pred_fp', [
    #(td1['wse1_rlayV_fp'], td1['wse1_rlay3_fp']),
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

#===============================================================================
# tests.points--------
#===============================================================================
 
@pytest.mark.parametrize('true_fp, pred_fp, sample_pts_fp', [
    (td1['wse1_rlayV_fp'], td1['wse1_rlay3_fp'], td1['sample_pts_fp']),
 
    ]) 
def test_get_samples(true_fp, pred_fp, sample_pts_fp, ses):
    gdf = ses.get_samples(true_fp=true_fp, pred_fp=pred_fp, sample_pts_fp=sample_pts_fp)
    
    #gdf.to_pickle(r'l:\09_REPOS\03_TOOLS\FloodDownscaler\tests\data\fred01\vali\samps_gdf_0109.pkl')



@pytest.mark.parametrize('samp_gdf_fp', [
    td1['samp_gdf_fp']
    ]) 
def test_get_samp_errs(samp_gdf_fp, ses):
    
    gdf = pd.read_pickle(samp_gdf_fp)
    
    ses.get_samp_errs(gdf)
    


@pytest.mark.parametrize('true_fp, pred_fp, sample_pts_fp', [
    (td1['wse1_rlayV_fp'], td1['wse1_rlay3_fp'], td1['sample_pts_fp']),
    #(wse1_rlayV_fp, wse1_rlay3_fp, None),
    ]) 
def test_run_vali_pts(true_fp, pred_fp, sample_pts_fp, ses):
    ses.run_vali_pts(sample_pts_fp, true_fp=true_fp, pred_fp=pred_fp)
    
    
    
    
#===============================================================================
# test.pipeline----
#===============================================================================

@pytest.mark.dev
@pytest.mark.parametrize('true_fp, pred_fp, sample_pts_fp, dem_fp', [
    (td1['wse1_rlayV_fp'], td1['wse1_rlay3_fp'], td1['sample_pts_fp'], td1['dem1_rlay_fp']),
    #(wse1_rlayV_fp, wse1_rlay3_fp, None),
    ]) 
def test_run_vali(true_fp, pred_fp, sample_pts_fp, dem_fp, ses):
    ses.run_vali(true_fp=true_fp, pred_fp=pred_fp, sample_pts_fp=sample_pts_fp, dem_fp=dem_fp)

    
 
@pytest.mark.parametrize('true_fp, pred_fp, sample_pts_fp, dem_fp', [
    (td1['wse1_rlayV_fp'], td1['wse1_rlay3_fp'], td1['sample_pts_fp'], td1['dem1_rlay_fp']),
    #(wse1_rlayV_fp, wse1_rlay3_fp, None),
    ]) 
def test_run_validator(true_fp, pred_fp, sample_pts_fp, dem_fp, tmp_dir):
    run_validator(true_fp, pred_fp, sample_pts_fp=sample_pts_fp, dem_fp=dem_fp, out_dir=tmp_dir)
    
    
    