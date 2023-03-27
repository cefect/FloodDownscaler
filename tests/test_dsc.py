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
#from fdsc.base import nicknames_d
from fdsc.control import run_downscale
from fdsc.control import Dsc_Session as Session
 
#from fdsc.bufferLoop import ar_buffer

from tests.conftest import (
    get_rlay_fp, crs_default, proj_lib,get_aoi_fp,par_algoMethodKwargs,
    par_method_kwargs,
 
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
def wrkr(init_kwargs, crs= crs_default):
    with Session(crs=crs, **init_kwargs) as session:
        yield session
 

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
    
    
#===============================================================================
# @pytest.mark.parametrize('dem_fp, wse_fp, crs', [
#     (dem1_rlay_fp, wse2_rlay_fp, crs_default),
#     (proj_lib['fred01']['dem1_rlay_fp'], proj_lib['fred01']['wse2_rlay_fp'], proj_lib['fred01']['crs'])
#     ]) 
# def test_p0(dem_fp, wse_fp, crs, tmp_path, wrkr):    
#     wrkr.p0_load_rasters(wse_fp, dem_fp, crs=crs, out_dir=tmp_path)
#     
#===============================================================================

@pytest.mark.parametrize('dem_fp, wse_fp', [
    (dem1_rlay_fp, wse2_rlay_fp),
    (proj_lib['fred01']['dem1_rlay_fp'], proj_lib['fred01']['wse2_rlay_fp'])
    ]) 
def test_p1(dem_fp, wse_fp, wrkr):    
    wrkr.p1_wetPartials(wse_fp, dem_fp)



#===============================================================================
# @pytest.mark.parametrize('dem_fp, wse_fp', [
#     (dem1_rlay_fp, wse1_rlay2_fp),
#     (proj_lib['fred01']['dem1_rlay_fp'], proj_lib['fred01']['wse1_rlay2_fp']), 
#     ]) 
# @pytest.mark.parametrize(*par_algoMethodKwargs)
# def test_p2(dem_fp, wse_fp, 
#             method, kwargs,
#             wrkr):
#     if method in ['Schumann14', 'none']: #skip those w/o phases
#         pass
#     else:
#         wrkr.p2_dryPartials(wse_fp, dem_fp, dryPartial_method=method, run_kwargs=kwargs)
#===============================================================================
    
#===============================================================================
# @pytest.mark.parametrize('dem_fp, wse_fp', [
#     (dem1_rlay_fp, wse1_rlay2_fp),
#     (proj_lib['fred01']['dem1_rlay_fp'], proj_lib['fred01']['wse1_rlay2_fp']),
#  
#     ])
# def test_p2_costGrowSimple(dem_fp, wse_fp, wrkr):
#     wrkr.run_costGrowSimple(wse_fp, dem_fp)
#===============================================================================


@pytest.mark.parametrize('wse_fp', [
    (wse1_rlay3_fp),
    (proj_lib['fred01']['wse1_rlay3_fp']),
 
    ])
def test_p2_filter_isolated(wse_fp, wrkr):
    wrkr._filter_isolated(wse_fp)
    

@pytest.mark.parametrize('dem_fp, wse_fp', [
    (dem1_rlay_fp, wse1_rlay2_fp),
    (proj_lib['fred01']['dem1_rlay_fp'], proj_lib['fred01']['wse1_rlay2_fp']), 
    ])
def test_p2_bufferGrow(dem_fp, wse_fp, wrkr):
    wrkr.get_bufferGrowLoop_DP(wse_fp, dem_fp, loop_range=range(5))
    

#===============================================================================
# @pytest.mark.parametrize('wse_ar',[
#     (wse1_ar2),
#     ]) 
# def test_ar_buffer(wse_ar):
#     ar_buffer(wse_ar)
#===============================================================================



@pytest.mark.parametrize('dem_fp, wse_fp', [
    (dem1_rlay_fp, wse2_rlay_fp),
    (proj_lib['fred01']['dem1_rlay_fp'], proj_lib['fred01']['wse2_rlay_fp'])
    ]) 
@pytest.mark.parametrize('backend', ['gr', 'rio'])
def test_schu14(dem_fp, wse_fp, wrkr, backend):
    wrkr.run_schu14(wse_fp, dem_fp, buffer_size=float(2/3), r2p_backend=backend)
    


@pytest.mark.parametrize('dem_fp, wse_fp', [
    (dem1_rlay_fp, wse2_rlay_fp),
    #(proj_lib['fred01']['dem1_rlay_fp'], proj_lib['fred01']['wse2_rlay_fp'])
    ])
@pytest.mark.parametrize(*par_algoMethodKwargs)
def test_runr(dem_fp, wse_fp, tmp_path, method, kwargs, logger):    
    run_downscale(dem_fp, wse_fp,  out_dir=tmp_path, run_name='test',logger=logger,
                  method=method, **kwargs)


@pytest.mark.dev
@pytest.mark.parametrize('dem_fp, wse_fp', [
    (dem1_rlay_fp, wse2_rlay_fp),
    #(proj_lib['fred01']['dem1_rlay_fp'], proj_lib['fred01']['wse2_rlay_fp'])
    ])
@pytest.mark.parametrize('method_pars', [par_method_kwargs])
def test_run_dsc_multi(dem_fp, wse_fp, method_pars, wrkr):
    wrkr.run_dsc_multi(dem_fp, wse_fp, method_pars=method_pars)
 
    
    
