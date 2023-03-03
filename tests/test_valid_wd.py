'''
Created on Mar. 3, 2023

@author: cefect
'''

#===============================================================================
# IMPORTS-------
#===============================================================================
import pytest, copy, os, random, re
xfail = pytest.mark.xfail


import numpy as np
import pandas as pd
import geopandas as gpd

from hp.rio import get_depth
from hp.tests.tools.rasters import get_poly_fp_from_rlay

from tests.conftest import (
      proj_lib, get_rlay_fp, crs_default
    )

from fdsc.analysis.valid.v_wd import ValidatePoints

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
def wrkr(logger, tmp_path):
    with ValidatePoints(logger=logger, 
                 ) as ses:
        yield ses

#===============================================================================
# tests.points--------
#===============================================================================
#convert  Fred WSE to depths
f1 = lambda x:get_depth(td1['dem1_rlay_fp'], x)
td1_wd_tp_fps = (f1(td1['wse1_rlayV_fp']), f1(td1['wse1_rlay3_fp']))
                
                 
                 
 

@pytest.mark.dev
@pytest.mark.parametrize('true_wd_fp, pred_wd_fp', [
    td1_wd_tp_fps,
    (wse1_rlayV_fp, wse1_rlay3_fp), 
    ]) 
def test_ValidatePoints_init(true_wd_fp, pred_wd_fp,
                         logger):
    """just the validation worker init"""
    
    with ValidatePoints(true_wd_fp=true_wd_fp, pred_wd_fp=pred_wd_fp,   logger=logger) as wrkr:
        pass
    
    

@pytest.mark.parametrize('true_fp, pred_inun_fp, sample_pts_fp', [
    (td1['wse1_rlayV_fp'], td1['wse1_rlay3_fp'], td1['sample_pts_fp']),
 
    ]) 
def test_get_samples(true_fp, pred_inun_fp, sample_pts_fp, ses):
    """TODO: use depth inputs"""
    gdf = ses.get_samples(true_fp=true_fp, pred_inun_fp=pred_inun_fp, sample_pts_fp=sample_pts_fp)
    
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