'''
Created on Apr. 1, 2023

@author: cefect
'''


import pytest, tempfile, datetime, os, copy, math
import numpy as np
import rasterio as rio

from hp.hyd import (
    write_poly_to_rlay, write_rlay_to_poly, get_inun_ar,HydTypes, get_wsh_ar, write_inun_rlay
    )
from definitions import src_dir

#===============================================================================
# toy data
#===============================================================================
from hp.tests.tools.rasters import get_rlay_fp
test_dir = os.path.join(src_dir, 'coms/hp/tests/data')
assert os.path.exists(test_dir), test_dir

from hp.tests.data.toy_rasters import proj_ar_d

#get a polygon from this
wse1_fp = get_rlay_fp(proj_ar_d['wse13'], 'wse13')

"""
write_inun_rlay(wse1_fp, 'WSE')
"""
inun_rlay_fp = os.path.join(test_dir, r'hyd\wse13_6x9_INUN.tif')

#inun_fp = write_rlay_to_poly(wse1_fp, dkey='WSE') #uniform
inun_poly_fp = os.path.join(test_dir, r'hyd\wse13_6x9_inun.geojson')
#inun_ar = get_inun_ar(wse1_mar, 'WSE')

dem_fp = get_rlay_fp(proj_ar_d['dem1'], 'dem1')

wsh_fp = get_rlay_fp(
    get_wsh_ar(proj_ar_d['dem1'], proj_ar_d['wse13']),
    'WSH1')

#===============================================================================
# tests--------
#===============================================================================
@pytest.mark.dev
@pytest.mark.parametrize('poly_fp',
                         [inun_poly_fp]
                         )
@pytest.mark.parametrize('rlay_ref', [wse1_fp])                        
def test_write_poly_to_rlay(poly_fp, rlay_ref):
    write_poly_to_rlay(poly_fp, rlay_ref=rlay_ref)
    

@pytest.mark.parametrize('fp, dkey', [
    (wse1_fp, 'WSE'),
    (dem_fp, 'DEM'),
    (wsh_fp, 'WSH'),
    (inun_poly_fp, 'INUN_POLY'),
    (inun_rlay_fp, 'INUN_RLAY'),
    
    ])
def test_load(fp, dkey):
    """loading tests apply_fp and assert_fp"""
    HydTypes(dkey).load_fp(fp)
    