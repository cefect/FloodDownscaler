'''
Created on Dec. 4, 2022

@author: cefect
'''
import pytest, os
import numpy as np
import pandas as pd
import rasterio as rio
import shapely.geometry as sgeo
from io import StringIO
import xarray as xr
import rioxarray
from pyproj.crs import CRS

crs_default = CRS.from_user_input(25832)
bbox_default = sgeo.box(0, 0, 60, 90)
 

#===============================================================================
# build test arrays
#===============================================================================
proj_lib=dict()

def get_ar_from_str(ar_str, dtype=float):
    return pd.read_csv(StringIO(ar_str), sep='\s+', header=None).astype(dtype).values

def get_wse_ar(ar_str, **kwargs):
    ar1 = get_ar_from_str(ar_str, **kwargs)
    return np.where(ar1==-9999, np.nan, ar1) #replace nans
    

#6x9
proj_lib['dem1'] = get_ar_from_str("""
    1    1    1    9    9    9
    1    1    1    9    9    9
    1    1    1    2    2    9
    2    2    2    9    2    9
    6    2    2    9    2    9
    2    2    2    9    2    9
    4    4    4    2    2    9
    4    4    4    9    9    9
    4    4    4    9    9    9
    """)

#2x3
proj_lib['wse1'] = get_wse_ar("""
    3    -9999
    4    -9999
    5    -9999    
    """) 
 
 

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
                      
                      
 
def get_rlay_fp(ar, layName, tmp_path, 
            crs = crs_default,
            bbox=bbox_default,
            ):
    assert isinstance(ar, np.ndarray)
    
    height, width  = ar.shape
    ofp = os.path.join(tmp_path, f'{layName}_{width}{height}.tif')
    
    
    
    #write
    with rio.open(ofp,'w',driver='GTiff',nodata=-9999,compress=None,
              height=height,width=width,count=1,dtype=ar.dtype,
            crs=crs,
            transform=rio.transform.from_bounds(*bbox.bounds,width, height),                
            ) as ds:
      
        ds.write(ar, indexes=1,masked=False)
        
        
    
        
    return ofp
    
#===============================================================================
# fixtuires
#===============================================================================
@pytest.fixture(scope='function')
def xds(proj_name):
    """retrieve componenet dataarrays by project name, then assemble a dataset"""
    return xr.Dataset({
        'dem':get_xda(proj_lib[proj_name]['dem']), 
        'wse':get_xda(proj_lib[proj_name]['wse'])})
     

 


    
    
