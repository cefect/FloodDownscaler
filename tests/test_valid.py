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

from fdsc.analysis.valid.v_ses import ValidateSession, run_validator

from tests.conftest import (
      proj_lib, get_rlay_fp, crs_default
    )

#===============================================================================
# test data-------
#===============================================================================
 
 
#===============================================================================
# fred data
#===============================================================================
td1 = proj_lib['fred01']
 
#convert  Fred WSE to depths
f = lambda wse_fp:get_depth(td1['dem1_rlay_fp'], wse_fp)
td1_wd1_rlayV_fp = f(td1['wse1_rlayV_fp'])
td1_wd1_rlay3_fp = f(td1['wse1_rlay3_fp'])

td1_wd_fps = (td1_wd1_rlayV_fp, td1_wd1_rlay3_fp, td1['sample_pts_fp'])
td1_inun1_fps=(td1['wse1_rlayV_fp'], td1['wse1_rlay3_fp'])
td1_inun2_fps=(td1['inun_vlay_fp'], td1['wse1_rlay3_fp'])
#===============================================================================
# toy data
#===============================================================================
from tests.data.toy import wse1_arV, wse1_ar3, dem1_ar

wse1_rlay3_fp = get_rlay_fp(wse1_ar3, 'wse13')
wse1_rlayV_fp = get_rlay_fp(wse1_arV, 'wse1V')
dem1_rlay_fp = get_rlay_fp(dem1_ar, 'dem1')
inun_poly_fp = get_poly_fp_from_rlay(wse1_rlayV_fp)

#convert to depths
f = lambda wse_fp:get_depth(dem1_rlay_fp, wse_fp)

toy_wd1_rlay3_fp = f(wse1_rlay3_fp)
toy_wd1_rlayV_fp = f(wse1_rlayV_fp)

toy_wd_fps = (toy_wd1_rlayV_fp, toy_wd1_rlay3_fp, None)
toy_inun1_fps = (wse1_rlayV_fp, wse1_rlay3_fp)
toy_inun2_fps = (inun_poly_fp, wse1_rlay3_fp)

#===============================================================================
# fixtures------------
#===============================================================================
 
@pytest.fixture(scope='function')
def ses(tmp_path, write, logger, test_name,
         crs=crs_default,
                    ):
    
    """Mock session for tests"""
 
    # np.random.seed(100)
    # random.seed(100)
    
    with ValidateSession(
                 # oop.Basic
                 out_dir=tmp_path,
                 tmp_dir=os.path.join(tmp_path, 'tmp_dir'),
                  proj_name='test',  # probably a better way to propagate through this key 
                 run_name=test_name[:8].replace('_', ''),
                 relative=True, write=write,  # avoid writing prep layers                 
                 logger=logger, overwrite=True,
                   
                   # oop.Session
                   logfile_duplicate=False,
                   
                   # RioSession
                   crs=crs,
                   ) as ses:
 
        yield ses
    
#===============================================================================
# test.pipeline----
#===============================================================================


@pytest.mark.parametrize('true_wd_fp, pred_wd_fp, sample_pts_fp', [
    td1_wd_fps,
    toy_wd_fps,
    ]) 
def test_run_vali_pts(true_wd_fp, pred_wd_fp, sample_pts_fp, ses):
    ses.run_vali_pts(sample_pts_fp, true_wd_fp=true_wd_fp, pred_wd_fp=pred_wd_fp)
    
@pytest.mark.dev
@pytest.mark.parametrize('true_inun_fp, pred_inun_fp', [
    td1_inun1_fps,
    td1_inun2_fps,
    toy_inun1_fps,
    toy_inun2_fps
    ]) 
def test_run_vali_inun(true_inun_fp, pred_inun_fp, ses):
    ses.run_vali_inun(true_inun_fp=true_inun_fp, pred_inun_fp=pred_inun_fp)
    

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
    
    
    
