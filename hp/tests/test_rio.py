'''
Created on Aug. 7, 2022

@author: cefect
'''


import pytest, tempfile, datetime, os, copy, math
import numpy as np
import rasterio as rio
from hp.rio import RioWrkr, write_array, write_resample, rlay_ar_apply
from rasterio.enums import Resampling
from pyproj.crs import CRS
from definitions import src_dir
 
#===============================================================================
# test data
#===============================================================================
from hp.tests.tools.rasters import get_rlay_fp, bbox_default, get_rand_ar

crsid = 2953
output_kwargs = dict(crs=rio.crs.CRS.from_epsg(crsid),
                     transform=rio.transform.from_origin(1,100,1,1)) 


#===============================================================================
# toy data
#===============================================================================
from hp.tests.data.toy_rasters import wse1_mar
 
wse1_mar_fp = get_rlay_fp(wse1_mar, 'wse1_toy_mar', crs=CRS.from_user_input(crsid), bbox=bbox_default)


#===============================================================================
# fixtures------
#===============================================================================
#===============================================================================
# @pytest.fixture(scope='function')
# def riowrkr(session, rlay_fp):
#     with RioWrkr(
#         rlay_ref_fp=rlay_fp,
#          # oop.Basic
#          session=session
#         
#         ) as wrkr:
#         yield wrkr
#===============================================================================
 

@pytest.fixture(scope='function')
def rlay_fp(ar, tmp_path): 
    return write_array(ar, os.path.join(tmp_path, 'ar.tif'), **output_kwargs)

#===============================================================================
# tests---------
#===============================================================================
#===============================================================================
# @pytest.mark.parametrize('ar', [np.random.random((3, 3))], indirect=False)
# def test_init(riowrkr, ar):    
#     assert isinstance(riowrkr.ref_name, str)
#===============================================================================

@pytest.mark.dev
@pytest.mark.parametrize('ar, scale', [
        (np.random.random((3, 3)), 3), #downsample
        (np.random.random((3, 3)), 1/3), #upsample
        (get_rand_ar((3,3)), 3),
        (get_rand_ar((3,5)), 2),
        (get_rand_ar((9,9)), 1/3),
    ], indirect=False)
@pytest.mark.parametrize('resampling', [
    Resampling.bilinear, Resampling.nearest])
def test_resample(resampling, scale, ar, rlay_fp):
    
    res_fp = write_resample(rlay_fp, resampling=resampling, scale=scale)
    
    #dataset = riowrkr.resample(resampling=resampling, scale=scale, write=True)
    
    #check
    def func(res_ar):
 
        assert isinstance(res_ar, np.ndarray)
        assert res_ar.shape==(int(ar.shape[0]*scale), int(ar.shape[1]*scale))
        
    rlay_ar_apply(res_fp, func)
    




#===============================================================================
# @pytest.mark.parametrize('rlay_ref_fp', [None], indirect=True)
# @pytest.mark.parametrize('test_rlay_fp_l', [test_rlay_fp_l]) 
# def test_merge(riowrkr, test_rlay_fp_l):
#  
#     
#     #load remainers
#     dsn_l = riowrkr._get_dsn(test_rlay_fp_l)
#     #dsn_l = [riowrkr.open_dataset(fp).name for fp in test_rlay_fp_l]
#  
#     ofp = riowrkr.merge(dsn_l)
#     
#     assert os.path.exists(ofp)
#===============================================================================
