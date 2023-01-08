'''
Created on Dec. 4, 2022

@author: cefect
'''

import pytest, copy, os, random, re
import numpy as np
import pandas as pd

import shapely.geometry as sgeo
 
xfail = pytest.mark.xfail

#from fdsc.scripts.disag import disag
from fdsc.scripts.dsc import run_downscale
from fdsc.scripts.dsc import Dsc_Session as Session

from tests.conftest import (
    get_xda, get_rlay_fp, crs_default, proj_lib,get_aoi_fp,
 
    )
 
#===============================================================================
# test data------
#===============================================================================
from tests.data.toy import dem1_ar, wse2_ar, wse1_ar2, wse1_ar3

dem1_rlay_fp = get_rlay_fp(dem1_ar, 'dem1') 
wse2_rlay_fp = get_rlay_fp(wse2_ar, 'wse2')
wse1_rlay2_fp = get_rlay_fp(wse1_ar2, 'wse12')
wse1_rlay3_fp = get_rlay_fp(wse1_ar3, 'wse13')
aoi_fp = get_aoi_fp(sgeo.box(0, 30, 60, 60))

#===============================================================================
# fixtures------------
#===============================================================================
@pytest.fixture(scope='function')
def wrkr(tmp_path,write,logger, test_name,
         crs= crs_default,
                    ):
    
    """Mock session for tests"""
 
    #np.random.seed(100)
    #random.seed(100)
    
    with Session(  
 
                 #oop.Basic
                 out_dir=tmp_path, 
                 tmp_dir=os.path.join(tmp_path, 'tmp_dir'),
                 #prec=prec,
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
# tests-------
#===============================================================================


@pytest.mark.parametrize('dem_fp, wse_fp, aoi_fp', [
    (dem1_rlay_fp, wse2_rlay_fp, aoi_fp),
    (proj_lib['fred01']['dem1_rlay_fp'], proj_lib['fred01']['wse2_rlay_fp'], proj_lib['fred01']['aoi_fp'])
    ]) 
def test_p0_clip(dem_fp, wse_fp, aoi_fp, tmp_path, wrkr): 
    wrkr._set_aoi(aoi_fp)
    wrkr.p0_clip_rasters(wse_fp, dem_fp, out_dir=tmp_path)
    
    
@pytest.mark.parametrize('dem_fp, wse_fp, crs', [
    (dem1_rlay_fp, wse2_rlay_fp, crs_default),
    (proj_lib['fred01']['dem1_rlay_fp'], proj_lib['fred01']['wse2_rlay_fp'], proj_lib['fred01']['crs'])
    ]) 
def test_p0(dem_fp, wse_fp, crs, tmp_path, wrkr):    
    wrkr.p0_load_rasters(wse_fp, dem_fp, crs=crs, out_dir=tmp_path)
    

@pytest.mark.parametrize('wse_ar, dem_ar, downscale', [
    (wse2_ar, dem1_ar, 3.0),
    ]) 
def test_p1(wse_ar, dem_ar, downscale, tmp_path, wrkr):    
    wrkr.p1_downscale_wetPartials(wse_ar, dem_ar,  downscale=downscale, out_dir=tmp_path)


@pytest.mark.parametrize('dem_fp, wse_fp', [
    (dem1_rlay_fp, wse1_rlay2_fp),
    (proj_lib['fred01']['dem1_rlay_fp'], proj_lib['fred01']['wse1_rlay2_fp']),
 
    ])
def test_p2_costGrowSimple(dem_fp, wse_fp, wrkr):
    wrkr.p2_dp_costGrowSimple(wse_fp, dem_fp)


@pytest.mark.parametrize('wse_fp', [
    (wse1_rlay3_fp),
    (proj_lib['fred01']['wse1_rlay3_fp']),
 
    ])
def test_p2_filter_isolated(wse_fp, wrkr):
    wrkr.filter_isolated(wse_fp)
    

@pytest.mark.dev
@pytest.mark.parametrize('dem_fp, wse_fp', [
    #(dem1_rlay_fp, wse2_rlay_fp),
    (proj_lib['fred01']['dem1_rlay_fp'], proj_lib['fred01']['wse2_rlay_fp'])
    ])
@pytest.mark.parametrize('dryPartial_method', [
    'costDistanceSimple','none'
    ])
def test_runr(dem_fp, wse_fp, tmp_path, dryPartial_method):    
    run_downscale(wse_fp, dem_fp, out_dir=tmp_path, run_name='test',
                  dryPartial_method=dryPartial_method)
    
