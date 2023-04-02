'''
Created on Feb. 4, 2023

@author: cefect

tools for testing rasters
'''
import pytest, tempfile, datetime, os, copy, math
from pyproj.crs import CRS
import rasterio as rio
import numpy as np
from io import StringIO
import shapely.geometry as sgeo
import numpy.ma as ma
import pandas as pd

from hp.rio import write_array
from hp.gpd import rlay_to_gdf
from hp.tests.conftest import temp_dir

nan, array = np.nan, np.array

#spatial defaults
from definitions import epsg,bbox 
crs_default = CRS.from_user_input(epsg)
bbox_default = sgeo.box(*bbox)


def get_rlay_fp(ar, layName, 
            ofp=None, 
            crs = crs_default,
            bbox=bbox_default,
            out_dir=None, 
            ):
    """simple raster writer for tests"""
    
    #===========================================================================
    # build out path
    #===========================================================================
    #assert isinstance(ar, np.ndarray)
    assert isinstance(ar, ma.MaskedArray)
    height, width  = ar.shape
    
    if out_dir is None:
        out_dir=temp_dir
        
    if not os.path.exists(out_dir):os.makedirs(out_dir)
    if ofp is None: 
        ofp = os.path.join(out_dir,f'{layName}_{width}x{height}.tif')
        
    return write_array(ar, ofp, crs=crs, 
                        transform=rio.transform.from_bounds(*bbox.bounds,width, height),
                        masked=True)
    
    
def get_mar(ar_raw):
    """convert a numpy array (w/ nulls) to rio mask-like"""
    return ma.array(
        np.where(np.isnan(ar_raw), -9999, ar_raw),
        mask=np.isnan(ar_raw), 
        fill_value=-9999,
        )

def get_ar_from_str(ar_str, dtype=float):
    return pd.read_csv(StringIO(ar_str), sep='\s+', header=None).astype(dtype).values

#===============================================================================
# def get_wse_ar(ar_str, **kwargs):
#     ar1 = get_ar_from_str(ar_str, **kwargs)
#     return np.where(ar1==-9999, np.nan, ar1) #replace nans
#===============================================================================


def get_rand_ar(shape, null_frac=0.1):
    ar_raw = np.random.random(shape)
    
    #add nulls randomly
    if null_frac>0:
        c = int(math.ceil(ar_raw.size*null_frac))
        ar_raw.ravel()[np.random.choice(ar_raw.size, c, replace=False)] = np.nan
        assert np.any(np.isnan(ar_raw))
        
    return ar_raw

def get_poly_fp_from_rlay(rlay_fp, convert_to_binary=True, ofp=None):
    """convert raster into polygon
    
    Parameters
    -----------
    convert_to_binary: bool, True
        polygonize the mask (rather than groups of data values)
        WARNING: only works on native masks
    
 
    
    """
 
    
    #geto geoDataFrame
    gdf = rlay_to_gdf(rlay_fp, convert_to_binary=convert_to_binary)
    
    #write
    if ofp is None:
        ofp = os.path.join(temp_dir, os.path.basename(rlay_fp).replace('.tif', '.gpkg'))
    gdf.to_file(ofp)
    
    return ofp
    
    