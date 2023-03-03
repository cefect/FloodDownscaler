'''
Created on Jan. 6, 2023

@author: cefect
'''

import pytest, copy, os, random, re
xfail = pytest.mark.xfail

import numpy as np
 

from hp.gpd import rlay_to_gdf
from hp.rio import get_depth
from hp.tests.tools.rasters import get_poly_fp_from_rlay

from fdsc.analysis.valid import ValidateMask, ValidatePoints, ValidateSession, run_validator

from tests.conftest import (
      proj_lib, get_rlay_fp, crs_default
    )

#===============================================================================
# test data-------
#===============================================================================
from tests.data.toy import wse1_arV, wse1_ar3, dem1_ar
 

td1 = proj_lib['fred01']

wse1_rlay3_fp = get_rlay_fp(wse1_ar3, 'wse13')
wse1_rlayV_fp = get_rlay_fp(wse1_arV, 'wse1V')
dem1_rlay_fp = get_rlay_fp(wse1_arV, 'dem1')

inun_poly_fp = get_poly_fp_from_rlay(wse1_rlayV_fp)
#===============================================================================
# fixtures------------
#===============================================================================
 
        

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
# test.pipeline----
#===============================================================================

@pytest.mark.dev
@pytest.mark.parametrize('true_wd_fp, pred_wd_fp, sample_pts_fp', [
    (*td1_fps, td1['sample_pts_fp']),
    (*toy_fps, None),
    ]) 
def test_run_vali_pts(true_fp, pred_fp, sample_pts_fp, ses):
    ses.run_vali_pts(sample_pts_fp, true_fp=true_fp, pred_fp=pred_fp)
    

@pytest.mark.parametrize('wse_true_fp, pred_fp, sample_pts_fp, dem_fp, inun_true_fp', [
    (td1['wse1_rlayV_fp'], td1['wse1_rlay3_fp'], td1['sample_pts_fp'], td1['dem1_rlay_fp'], td1['inun_vlay_fp']),
    #(wse1_rlayV_fp, wse1_rlay3_fp, None),
    ]) 
def test_run_vali(wse_true_fp, pred_fp, sample_pts_fp, dem_fp, inun_true_fp, ses):
    ses.run_vali(wse_true_fp=wse_true_fp, inun_true_fp=inun_true_fp,
                 pred_fp=pred_fp, sample_pts_fp=sample_pts_fp, dem_fp=dem_fp)

    

@pytest.mark.parametrize('true_fp, pred_fp, sample_pts_fp, dem_fp', [
    #(td1['wse1_rlayV_fp'], td1['wse1_rlay3_fp'], td1['sample_pts_fp'], td1['dem1_rlay_fp']),
    (inun_poly_fp, wse1_rlay3_fp, None, dem1_rlay_fp),
    ]) 
def test_run_validator(true_fp, pred_fp, sample_pts_fp, dem_fp, tmp_path):
    run_validator(true_fp, pred_fp, sample_pts_fp=sample_pts_fp, dem_fp=dem_fp, out_dir=tmp_path)
    
    
    
