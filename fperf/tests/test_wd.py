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

#from hp.rio import get_depth
from hp.oop import Session
from hp.hyd import get_wsh_rlay
from hp.tests.tools.rasters import get_rlay_fp
from fperf.tests.conftest import proj_lib
 

from fperf.wd import ValidatePoints

#===============================================================================
# test data-------
#===============================================================================

 
#===============================================================================
# fred data
#===============================================================================
td1 = proj_lib['fred01']

#convert  Fred WSE to depths
f = lambda wse_fp:get_wsh_rlay(td1['dem1_rlay_fp'], wse_fp)
td1_wd1_rlayV_fp = f(td1['wse1_rlayV_fp'])
td1_wd1_rlay3_fp = f(td1['wse1_rlay3_fp'])

td1_fps = (td1_wd1_rlayV_fp, td1_wd1_rlay3_fp)
#===============================================================================
# toy data
#===============================================================================
from fperf.tests.data.toy import wse1_arV, wse1_ar3, dem1_ar, crs_default, bbox_default

gfp = lambda ar, name:get_rlay_fp(ar, name, crs=crs_default, bbox=bbox_default)
wse1_rlay3_fp = gfp(wse1_ar3, 'wse13')
wse1_rlayV_fp = gfp(wse1_arV, 'wse1V')
dem1_rlay_fp = gfp(dem1_ar, 'dem1')

#convert to depths
f = lambda wse_fp:get_wsh_rlay(dem1_rlay_fp, wse_fp)

toy_wd1_rlay3_fp = f(wse1_rlay3_fp)
toy_wd1_rlayV_fp = f(wse1_rlayV_fp)

toy_fps = toy_wd1_rlayV_fp, toy_wd1_rlay3_fp

#===============================================================================
# fixtures------------
#===============================================================================
class ValidateSessionTester(ValidatePoints, Session):
    pass
         
@pytest.fixture(scope='function')
def wrkr(init_kwargs):    
    """Mock session for tests""" 
    with ValidateSessionTester(**init_kwargs) as ses: 
        yield ses

#===============================================================================
# tests.points--------
#===============================================================================
 
 
    
    

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
    

