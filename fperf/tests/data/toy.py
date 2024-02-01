'''
Created on Dec. 5, 2022

@author: cefect

toy test data
'''
import pytest
import numpy as np
import pandas as pd
import numpy.ma as ma
import shapely.geometry as sgeo
from pyproj.crs import CRS
 
nan, array = np.nan, np.array

"""setup to construct when called
seems cleaner then building each time we init (some tests dont need these)
more seamless with using real data in tests"""

#===============================================================================
# helpers
#===============================================================================
from hp.tests.tools.rasters import (
    get_mar, get_ar_from_str, get_rlay_fp,crs_default, 
    )
#from hp.hyd import get_wsh_ar
from hp.np import apply_block_reduce2, get_support_ratio



#===============================================================================
# raw data
#===============================================================================
dem1_ar = get_mar(
    get_ar_from_str("""
    1    1    1    9    9    9    
    1    1    1    9    9    9
    1    1    1    2    2    9
    2    2    2    9    2    9
    6    2    2    9    2    9
    2    2    2    9    2    9
    4    4    4    2    2    9
    4    4    4    9    9    9
    4    4    4    9    1    1
    """))


wse2_ar = get_mar(  #get_mar converts 
    array([
        [3.0, nan],
        [4.0, nan],
        [5.0, nan]
        ])
    )

"""dummy validation against wse1_ar3
1FP, 1FN"""
wse1_arV = get_mar(
    array([
        [ 3.,  3.,  3., nan, nan, nan],
       [ 3.,  3.,  3., nan, nan, nan],
       [ 3.,  3.,  3.,  3.,  3., nan],
       [ 4.,  4.,  4., nan,  3., nan],
       [nan,  4.,  4., nan,  3., nan],
       [ 4.,  4.,  4., nan,  3., nan],
       [ 5.,  5.,  5.,  5.,  5., nan],
       [ 5.,  5.,  5., nan, nan, 5.],
       [ 5.,  5.,  5., nan, nan, nan]])
    )

aoi_box = sgeo.box(0, 0, 6, 9)


#===============================================================================
# intermittent data
#===============================================================================
"""p1_downscale_wetPartials"""
wse1_ar2 =get_mar(
    np.array([
        [ 3.,  3.,  3., np.nan, np.nan, np.nan],
       [ 3.,  3.,  3., np.nan, np.nan, np.nan],
       [ 3.,  3.,  3., np.nan, np.nan, np.nan],
       [ 4.,  4.,  4., np.nan, np.nan, np.nan],
       [nan,  4.,  4., np.nan, np.nan, np.nan],
       [ 4.,  4.,  4., np.nan, np.nan, np.nan],
       [ 5.,  5.,  5., np.nan, np.nan, np.nan],
       [ 5.,  5.,  5., np.nan, np.nan, np.nan],
       [ 5.,  5.,  5., np.nan, np.nan, np.nan]])
    )

"""phase2: _null_dem_violators"""
wse1_ar3 = get_mar(
    array([
        [ 3.,  3.,  3., nan, nan, nan],
       [ 3.,  3.,  3., nan, nan, nan],
       [ 3.,  3.,  3.,  3.,  3., nan],
       [ 4.,  4.,  4., nan,  4., nan],
       [nan,  4.,  4., nan,  4., nan],
       [ 4.,  4.,  4., nan,  4., nan],
       [ 5.,  5.,  5.,  5.,  5., nan],
       [ 5.,  5.,  5., nan, nan, nan],
       [ 5.,  5.,  5., nan, nan,  5.]])
    )

#===============================================================================
# computed data
#===============================================================================
s = dem1_ar.shape
bbox_default = sgeo.box(0, 0, s[1], s[0]) #(0.0, 0.0, 6.0, 12.0)
#print(f'toy bbox_default={bbox_default.bounds}')

#aggregated DEM
aggscale=get_support_ratio(dem1_ar, wse2_ar)
dem2_ar = apply_block_reduce2(dem1_ar, aggscale=int(aggscale), func=np.mean)

proj_ar_d = {
    'wse13':wse1_ar3, 'wse1V':wse1_arV, 'wse2':wse2_ar, 'dem1':dem1_ar, 'dem2':dem2_ar,
    }
#===============================================================================
# fixtures
#===============================================================================
"""not sure how this would work
because we want to specify combinations of layers by names as inputs
    which use different names within the test
    
would have to be a fixture of a fixture?
"""
@pytest.fixture(scope='session')
def toy_rlay_fp(layerName, crs=crs_default, bbox=bbox_default):
    ar = proj_ar_d[layerName]
    return get_rlay_fp(ar, layerName, crs=crs, bbox=bbox)



