'''
Created on Dec. 4, 2022

@author: cefect
'''
import pytest, os, tempfile, datetime
import numpy as np
import pandas as pd
import rasterio as rio
import shapely.geometry as sgeo

import xarray as xr
import rioxarray
from pyproj.crs import CRS
from definitions import src_dir


from hp.logr import get_new_console_logger, logging

crs_default = CRS.from_user_input(25832)
bbox_default = sgeo.box(0, 0, 60, 90)
 

#===============================================================================
# setup test arrays
#===============================================================================

proj_lib = dict()
proj_lib['fred01'] = {
    'wse2_rlay_fp':os.path.join(src_dir, r'tests/data/fred01/testr_test00_0806_fwse.tif'),
    
    #p1_downscale_wetPartials
    'wse1_rlay2_fp':os.path.join(src_dir, r'tests/data/fred01/wse1_ar2.tif'),
    
    #p2_dp_costGrowSimple._filter_dem_violators
    'wse1_rlay3_fp':os.path.join(src_dir, r'tests/data/fred01/wse1_ar3.tif'),
        
    'dem1_rlay_fp':os.path.join(src_dir, r'tests\data\fred01\dem.tif'),
    }


 
 
 

#===============================================================================
# helpers
#===============================================================================

def get_xy_coords(transform, shape):
    """return an array of spatial values for x and y
    
    surprised there is no builtin
    
    this is needed  by xarray
    
    print(f'x, cols:{s[1]}    y, rows:{s[0]}')
    """
    transformer = rio.transform.AffineTransformer(transform) 
    x_ar, _ = transformer.xy(np.full(shape[1], 0), np.arange(shape[1])) #rows, cols            
    _, y_ar = transformer.xy(np.arange(shape[0]), np.full(shape[0], 0)) #rows, cols
    
    return x_ar, y_ar

 
def get_xda(ar,
            transform=None, 
            #transform=rio.transform.from_origin(0,0,1,1), 
            crs = crs_default,            
            ):
    """build a rioxarray from scratch"""
    
    if transform is None:
        transform = rio.transform.from_bounds(*bbox_default.bounds,ar.shape[1], ar.shape[0])
    
    xs, ys = get_xy_coords(transform, ar.shape)            
 
    return xr.DataArray(np.array([ar]), 
                       coords={'band':[1],  'y':ys, 'x':xs} #order is important
                       #coords = [[1],  ys, xs,], dims=["band",  'y', 'x']
                       #).rio.write_transform(transform_i
                       ).rio.write_nodata(-9999, inplace=True
                      ).rio.set_crs(crs, inplace=True)    
                      
                      
 
def get_rlay_fp(ar, layName, 
            ofp=None, 
            crs = crs_default,
            bbox=bbox_default,
            ):
    
    assert isinstance(ar, np.ndarray)
    height, width  = ar.shape
    
    if ofp is None:
        out_dir = os.path.join(tempfile.gettempdir(), __name__, datetime.datetime.now().strftime('%Y%m%d'))
        if not os.path.exists(out_dir):os.makedirs(out_dir)
        ofp = os.path.join(out_dir,f'{layName}_{width}{height}.tif')
    
    
 
    
    
    
    #write
    with rio.open(ofp,'w',driver='GTiff',nodata=-9999,compress=None,
              height=height,width=width,count=1,dtype=ar.dtype,
            crs=crs,
            transform=rio.transform.from_bounds(*bbox.bounds,width, height),                
            ) as ds:
      
        ds.write(ar, indexes=1,masked=False)
        
    print(f'wrote {ar.shape} to {ofp}')
    
        
    return ofp
    
#===============================================================================
# MISC----
#===============================================================================
@pytest.fixture(scope='session')
def write():
    write=False
    if write:
        print('WARNING!!! runnig in write mode')
    return write

@pytest.fixture(scope='function')
def test_name(request):
    return request.node.name.replace('[','_').replace(']', '_')

@pytest.fixture(scope='session')
def logger():
    return get_new_console_logger(level=logging.DEBUG)

#===============================================================================
# fixtuires
#===============================================================================
@pytest.fixture(scope='function')
def xds(proj_name):
    """retrieve componenet dataarrays by project name, then assemble a dataset"""
    return xr.Dataset({
        'dem':get_xda(proj_lib[proj_name]['dem']), 
        'wse':get_xda(proj_lib[proj_name]['wse'])})
     

 


    
    
