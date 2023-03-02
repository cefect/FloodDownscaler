'''
Created on Mar. 2, 2023

@author: cefect
'''
import pytest, tempfile, datetime, os, copy, math
import numpy as np
from hp.rio import RioWrkr, write_array, rio
from rasterio.enums import Resampling
from definitions import src_dir
from pyproj.crs import CRS



from hp.gpd import write_rlay_to_polygon
#===============================================================================
# test data
#===============================================================================
from hp.tests.tools.rasters import get_rlay_fp, get_wse_ar, bbox_default, get_rand_ar, crs_default

from hp.tests.data.toy_rasters import wse1_mar
 
wse1_mar_fp = get_rlay_fp(wse1_mar, 'wse1_toy_mar', crs=crs_default, bbox=bbox_default)


#===============================================================================
# tests
#===============================================================================


@pytest.mark.parametrize('rlay_fp', [wse1_mar_fp])
def test_rlay_to_polygon(rlay_fp):
    write_rlay_to_polygon(rlay_fp)