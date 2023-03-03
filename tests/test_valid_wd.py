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

 
#===============================================================================
# fred data
#===============================================================================
td1 = proj_lib['fred01']

#convert  Fred WSE to depths
f = lambda wse_fp:get_depth(td1['dem1_rlay_fp'], wse_fp)
td1_wd1_rlayV_fp = f(td1['wse1_rlayV_fp'])
td1_wd1_rlay3_fp = f(td1['wse1_rlay3_fp'])

td1_fps = (td1_wd1_rlayV_fp, td1_wd1_rlay3_fp)
#===============================================================================
# toy data
#===============================================================================
from tests.data.toy import wse1_arV, wse1_ar3, dem1_ar

wse1_rlay3_fp = get_rlay_fp(wse1_ar3, 'wse13')
wse1_rlayV_fp = get_rlay_fp(wse1_arV, 'wse1V')
dem1_rlay_fp = get_rlay_fp(dem1_ar, 'dem1')

#convert to depths
f = lambda wse_fp:get_depth(dem1_rlay_fp, wse_fp)

toy_wd1_rlay3_fp = f(wse1_rlay3_fp)
toy_wd1_rlayV_fp = f(wse1_rlayV_fp)

toy_fps = toy_wd1_rlayV_fp, toy_wd1_rlay3_fp

#===============================================================================
# fixtures------------
#===============================================================================
@pytest.fixture(scope='function')
def wrkr(logger, tmp_path, test_name):
    with ValidatePoints(
                 out_dir=tmp_path, 
                 tmp_dir=os.path.join(tmp_path, 'tmp_dir'),
                  proj_name='test', #probably a better way to propagate through this key 
                 run_name=test_name[:8].replace('_',''), fancy_name='fancy_name',
                 ) as ses:
        yield ses

#===============================================================================
# tests.points--------
#===============================================================================
 


@pytest.mark.parametrize('true_wd_fp, pred_wd_fp', [
    td1_fps,
    toy_fps, 
    ]) 
def test_ValidatePoints_init(true_wd_fp, pred_wd_fp,
                         logger):
    """just the validation worker init"""
    
    with ValidatePoints(true_wd_fp=true_wd_fp, pred_wd_fp=pred_wd_fp,   logger=logger) as wrkr:
        pass
    
    

@pytest.mark.parametrize('true_wd_fp, pred_wd_fp, sample_pts_fp', [
    (*td1_fps, td1['sample_pts_fp']), 
    ])
def test_get_samples(true_wd_fp, pred_wd_fp, sample_pts_fp, wrkr):
 
    gdf = wrkr.get_samples(true_wd_fp=true_wd_fp, pred_wd_fp=pred_wd_fp, sample_pts_fp=sample_pts_fp)
    
    #gdf.to_pickle(r'l:\09_REPOS\03_TOOLS\FloodDownscaler\tests\data\fred01\vali\samps_gdf_0109.pkl')



@pytest.mark.parametrize('samp_gdf_fp', [
    td1['samp_gdf_fp']
    ]) 
def test_get_samp_errs(samp_gdf_fp, wrkr):
    
    gdf = pd.read_pickle(samp_gdf_fp)
    
    wrkr.get_samp_errs(gdf)
    

