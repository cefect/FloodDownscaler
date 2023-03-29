'''
Created on Feb. 4, 2023

@author: cefect

testing varios pixels to points functions

PERFORMANCE TESTS
-----------------
for rlay big (1e3x1e3) and max_workers=os.cpu_count()
    test_gpd: ~40secs
    test_georasters: 0.8secs
    test_rio: ~10secs
    
    
'''
import pytest, tempfile, datetime, os, copy, math
from pyproj.crs import CRS
import shapely.geometry as sgeo
import numpy.ma as ma
import numpy as np
import geopandas as gpd


from hp.tests.tools.rasters import get_rlay_fp, get_mar


 
#===============================================================================
# parmeters
#===============================================================================
temp_dir = os.path.join(tempfile.gettempdir(), __name__, datetime.datetime.now().strftime('%Y%m%d'))
crs_default = CRS.from_user_input(25832)
bbox_default = sgeo.box(0, 0, 100, 100)

#===============================================================================
# test data
#===============================================================================
 
rlay_big_fp = get_rlay_fp(get_mar(np.random.random((int(1e3), int(1e3)))), 'rlay_big', out_dir=temp_dir)

rlay_small_fp = get_rlay_fp(get_mar(np.random.random((int(1e2), int(1e2)))), 'rlay_small', out_dir=temp_dir)
#===============================================================================
# fixtures
#===============================================================================

#===============================================================================
# tests--------
#===============================================================================
from hp.rio_to_points import raster_to_points_simple as gpd_func

@pytest.mark.parametrize('fp', [rlay_small_fp, 
                                #rlay_big_fp #12secs
                                ], indirect=False)
@pytest.mark.parametrize('drop_mask', [False], indirect=False)
@pytest.mark.parametrize('max_workers', [1, 4, None], indirect=False)
def test_gpd(fp, drop_mask, max_workers):
    result = gpd_func(fp, drop_mask=drop_mask, max_workers=max_workers)
    
    assert isinstance(result, gpd.GeoSeries)

 
from hp.gr import pixels_to_points as gr_func


@pytest.mark.dev
@pytest.mark.parametrize('fp', [rlay_small_fp,
                                 #rlay_big_fp #65 secs
                                 ], indirect=False)
def test_georasters(fp):
    gr_func(fp)
    
    
 
 
    



