'''
Created on Dec. 4, 2022

@author: cefect
'''

import pytest, copy, os, random, re
import numpy as np
import pandas as pd


import xarray as xr
xfail = pytest.mark.xfail

from fdsc.scripts.basic import disag
from fdsc.scripts.coms import run_downscale

from tests.conftest import proj_lib, get_xda, get_rlay_fp, crs_default

import matplotlib.pyplot as plt
#create a raster array from scratch
#calculate the sptial dimensions


 

#===============================================================================
# tests-------
#===============================================================================
@pytest.mark.dev
@pytest.mark.parametrize('dem_ar, wse_ar', [
    (proj_lib['dem1'], proj_lib['wse1'])
    ])
def test_runr(dem_ar, wse_ar, tmp_path):
    
    #build rlays
    wse_fp = get_rlay_fp(wse_ar, 'wse', tmp_path)
    dem_fp = get_rlay_fp(dem_ar, 'dem', tmp_path)
    
    
    run_downscale(wse_fp, dem_fp, crs=crs_default)

 
@pytest.mark.parametrize('dem_ar, wse_ar', [
    (proj_lib['dem1'], proj_lib['wse1'])
    ])
def test_xar(dem_ar, wse_ar):
 
    #build a dataset from the dataarrays
    dem_ds = get_xda(dem_ar)
    wse_ds = get_xda(wse_ar)
 
 
    
    """
    xds['dem'].plot()
    xds['wse'].plot()
    plt.show()
    """
 


#===============================================================================
# @pytest.mark.parametrize('ar', [
#     np.arange(4*4).reshape(4,4),
#     ])
# @pytest.mark.parametrize('scale', [2])
# def test_disag_ar(ar, scale):
#     disag(ar, downscale=scale)
#===============================================================================
    