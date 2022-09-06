'''
Created on Aug. 7, 2022

@author: cefect
'''


import pytest, tempfile, datetime, os, copy, math
import numpy as np
from hp.rio import RioWrkr, write_array, rio
from rasterio.enums import Resampling
from definitions import src_dir
 

 

crsid = 2953
output_kwargs = dict(crs=rio.crs.CRS.from_epsg(crsid),
                     transform=rio.transform.from_origin(1,100,1,1)) 

def get_rand_ar(shape, null_frac=0.1):
    ar_raw = np.random.random(shape)
    
    #add nulls randomly
    if null_frac>0:
        c = int(math.ceil(ar_raw.size*null_frac))
        ar_raw.ravel()[np.random.choice(ar_raw.size, c, replace=False)] = np.nan
        assert np.any(np.isnan(ar_raw))
        
    return ar_raw
#===============================================================================
# fixtures------
#===============================================================================
@pytest.fixture(scope='function')
def riowrkr(session, rlay_fp):
    with RioWrkr(
        rlay_ref_fp=rlay_fp,
         #oop.Basic
         session=session
        
        ) as wrkr:
        yield wrkr
        
 

@pytest.fixture(scope='function')
def rlay_fp(ar, tmp_path): 
    return write_array(ar, os.path.join(tmp_path, 'ar.tif'), **output_kwargs)

#===============================================================================
# tests---------
#===============================================================================
@pytest.mark.parametrize('ar', [np.random.random((3, 3))], indirect=False)
def test_init(riowrkr, ar):    
    assert isinstance(riowrkr.ref_name, str)

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
def test_resample(riowrkr, resampling, scale, ar, rlay_fp):
    
    dataset = riowrkr.resample(resampling=resampling, scale=scale, write=True)
    
    ar_res =  dataset.read(1)
    assert isinstance(ar_res, np.ndarray)
    assert ar_res.shape==(int(ar.shape[0]*scale), int(ar.shape[1]*scale))
    
 
 


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