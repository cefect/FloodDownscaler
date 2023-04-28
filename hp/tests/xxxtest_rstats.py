'''
Created on Sep. 17, 2022

@author: cefect
'''
import pytest, copy, os, random
import rasterio as rio
import numpy as np
from hp.rio import write_array
import shapely.geometry as sgeo
from pyproj.crs import CRS
import geopandas as gpd
import hp.rstats
#===============================================================================
# globals
#===============================================================================
bbox_base = sgeo.box(0, 0, 100, 100)
crs=CRS.from_user_input(2953)
finv_fp = r'C:\LS\09_REPOS\01_COMMON\coms\hp\tests\data\finv_SJ_test_0906.geojson'
#===============================================================================
# fixtures
#===============================================================================
@pytest.fixture(scope='function')
def rlay_fp(ar_raw, tmp_path, shape):
    ofp = os.path.join(tmp_path, 'array_%i.tif'%ar_raw.size)
    
    width, height = ar_raw.shape
    
    write_array(ar_raw, ofp, crs=crs,
                 transform=rio.transform.from_bounds(*bbox_base.bounds,width, height),  
                 masked=False)
    
    return ofp
 
    
 
@pytest.fixture(scope='function')    
def ar_raw(shape):
    return np.random.random(shape)
    #return np.random.choice(np.array(list(cm_int_d.values())), size=shape)

 
#===============================================================================
# tests
#===============================================================================

@pytest.mark.parametrize('feats_fp', [finv_fp])
@pytest.mark.parametrize('shape', [(10,10)], indirect=False)
@pytest.mark.parametrize('cores', [1, None])
@pytest.mark.parametrize('kwargs', [dict(), 
                                    dict(stats=['count']
                                         )])
def test_01_rsamp(rlay_fp, feats_fp, cores, kwargs): 
    gdf = gpd.read_file(finv_fp, bbox=bbox_base).rename_axis('fid')
    hp.rstats.zonal_stats_multi(rlay_fp, gdf.geometry.values.tolist(), cores=cores, **kwargs)