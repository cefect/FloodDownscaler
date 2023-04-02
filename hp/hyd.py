'''
Created on Mar. 18, 2023

@author: cefect

special hydraulic helpers

Recognized Data Types
------------------
WSH: raster. 
    0:dry, >0:wet
    no mask
WSE: raster.
    masked=dry, nomask=wet
    partial mask
INUN_RLAY: raster
    0:dry, 1:wet
    no mask
INUN_POLY: geojson
    outside:dry, inside:wet
    
    
TODO
--------
make this a class object, where you can specify the data type

'''

import datetime, os, tempfile
import pandas as pd
import numpy as np
import numpy.ma as ma
import rasterio as rio
import rasterio.features
import shapely.geometry as sgeo
import geopandas as gpd
 

from hp.rio import (
    assert_rlay_simple, get_stats, assert_spatial_equal, get_ds_attr, write_array2, 
    load_array, get_profile, is_raster_file, rlay_ar_apply
    )

from hp.riom import (
    write_array_mask, _dataset_to_mar, assert_mask_ar, rlay_mar_apply, load_mask_array,
    write_extract_mask, assert_masked_ar
    )



#===============================================================================
# RASTERS CONVSERIONS -------
#===============================================================================
def get_wsh_rlay(dem_fp, wse_fp, out_dir = None, ofp=None):
    """add dem and wse to get a depth grid"""
    
    assert_spatial_equal(dem_fp, wse_fp)
    assert_type_fp(wse_fp, 'WSE')
    assert_type_fp(dem_fp, 'DEM')
    
    if ofp is None:
        if out_dir is None:
            out_dir = tempfile.gettempdir()
        if not os.path.exists(out_dir):os.makedirs(out_dir)
        
        fname = os.path.splitext( os.path.basename(wse_fp))[0] + '_wsh.tif'
        ofp = os.path.join(out_dir,fname)
    
    #===========================================================================
    # load
    #===========================================================================
    dem_ar = load_array(dem_fp, masked=True)
    
    wse_ar = load_array(wse_fp, masked=True)
    
    #===========================================================================
    # build raster
    #===========================================================================
    wd2M_ar = get_wsh_ar(dem_ar, wse_ar)
    
    #===========================================================================
    # write
    #===========================================================================
    return write_array2(wd2M_ar, ofp, masked=False, **get_profile(wse_fp))

def get_wsh_ar(dem_ar, wse_ar):
    
    assert_dem_ar(dem_ar)
    assert_wse_ar(wse_ar)
    
    #simple subtract
    wd_ar1 = wse_ar-dem_ar
    
    #filter dry    
    wd_ar2 = np.where(wd_ar1.mask, 0.0, wd_ar1.data)
 
    
    #filter negatives
    wd_ar3 = ma.array(
        np.where(wd_ar2<0.0, 0.0, wd_ar2.data),
        mask=np.full(wd_ar1.shape, False),
        fill_value=-9999)
    
    assert_wsh_ar(wd_ar3)
    
    return wd_ar3
        
    


def _get_ofp(fp, out_dir, name='wse', ext='tif'):
    """shortcut to get filepaths"""
    if out_dir is None:
        out_dir = tempfile.gettempdir()
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    fname = os.path.splitext(os.path.basename(fp))[0] + f'_{name}.{ext}'
    ofp = os.path.join(out_dir, fname)
    return ofp

def get_wse_rlay(dem_fp, wd_fp, out_dir = None, ofp=None):
    """add dem and wse to get a depth grid"""
    
    assert_spatial_equal(dem_fp, wd_fp)
    
    if ofp is None:
        ofp = _get_ofp(wd_fp, out_dir)
    
    #===========================================================================
    # load
    #===========================================================================
    dem_ar = load_array(dem_fp, masked=True)
    
        
    wd_ar = load_array(wd_fp, masked=True)
    
    wse_ar = get_wse_ar(dem_ar, wd_ar)
    
    #===========================================================================
    # write
    #===========================================================================
    return write_array2(wse_ar, ofp, masked=False, **get_profile(wd_fp))
    
    
    
def get_wse_ar(dem_ar, wd_ar):
    assert_dem_ar(dem_ar)
    assert_wsh_ar(wd_ar)
    #===========================================================================
    # add
    #===========================================================================
    wse_ar1 = wd_ar+dem_ar
    
    wse_ar2 = ma.array(
                wse_ar1.data, 
                 mask=wd_ar<=0.0, 
                 fill_value=dem_ar.fill_value)
    
    assert_wse_ar(wse_ar2)
    
    return wse_ar2
    
def get_inun_ar(ar_raw, dkey):
    """convert flood like array to inundation (wet=True)"""
    if dkey in ['INUN_RLAY']:
        ar = ar_raw
    elif dkey == 'WSE':
        ar = np.invert(ar_raw.mask)
    else:
        raise NotImplementedError(dkey)
        ar = ar.mask
        
    assert_inun_ar(ar, msg=dkey)
    
    return ar

#===============================================================================
# RASTER-POLY conversions -----------
#===============================================================================

def rlay_to_poly(rlay_fp, dkey,                    
                    window=None,
                    ):
    """build an inundation polygon from the rlay
    
    
    Pars
    ---------
    rlay_fp: str
        flood like grid
            WSH: 
    
    see also hp.rio.rlay_to_polygons
    """
    assert is_raster_file(rlay_fp)
    assert_type_fp(rlay_fp, dkey)
    
    #===========================================================================
    # collect polygons
    #===========================================================================
    
    with rio.open(rlay_fp, mode='r') as dataset:
        #load the array by type
        ar_raw = _get_hyd_ar(dataset, dkey, window=window)
        
        #convert to mask (wet=True)
        ar_bool = get_inun_ar(ar_raw, dkey)
 
        #convert to binary for rio (1=wet)
        ar_binary = np.where(ar_bool, 1, 0)
 
        #mask = image != src.nodata
        geo_d=dict()
        for geom, val in rasterio.features.shapes(ar_binary, mask=~ar_bool, 
                                                  transform=dataset.transform,
                                                  connectivity=8):
            
            geo_d[val] = sgeo.shape(geom)
    
    assert len(geo_d)==1
    assert val==0
    
    return geo_d[val]

def polyVlay_to_ar(poly_fp,
                       rlay_ref=None,
                       out_shape=None,
                       transform=None,
                   crs=None,

                            **kwargs):
    """convert an inundation polygon to a boolean inundation raster
    1=wet
    """
    assert_type_fp(poly_fp, 'INUN_POLY')
    

    
 
    
    #===========================================================================
    # load poolygon
    #===========================================================================
    gdf = gpd.read_file(poly_fp)

    if not crs is None:
        gdf = gdf.to_crs(crs)
    
    #===========================================================================
    # get ref values
    #===========================================================================
    if out_shape is None or transform is None:
        assert os.path.exists(rlay_ref)
        with rasterio.open(rlay_ref, 'r+',**kwargs) as src:
            if not crs is None:
                src.crs=crs
            assert src.crs.to_epsg()==gdf.crs.to_epsg(), f'crs mismatch'
            """
            type(gdf.crs)
            """
            out_shape = src.shape
            transform=src.transform


    

    #===========================================================================
    # # Generate a mask from the geojson geometry
    #===========================================================================
    mask_ar = rasterio.features.geometry_mask(gdf.geometry, 
                                   out_shape=out_shape, transform=transform,
                                   invert=True)
    
    assert_inun_ar(mask_ar)
    
    return mask_ar

#===============================================================================
# WRITERS-----------
#===============================================================================
def write_wsh_boolean(fp,
                 ofp=None, out_dir=None,
                 load_kwargs=dict(),
                 ):
    """write a boolean (0,1) raster of the inundation represented by the input WSH"""
    if ofp is None:
        ofp = _get_ofp(fp, out_dir, name='inun')
        
    #load the raw
    mar_raw = load_array(fp, **load_kwargs)
    assert_wsh_ar(mar_raw)
 
    #write mask True=dry, False=wet
    return write_array_mask(mar_raw.data==0, ofp=ofp, maskType='binary',**get_profile(fp))

def write_inun_rlay(fp, dkey,
                    ofp=None, out_dir=None,
                    **kwargs):
    """write a boolean inundation ratser from a WSE or WSH layer"""
    
    assert_type_fp(fp, dkey)
    
    if ofp is None:
        ofp = _get_ofp(fp, out_dir, name='INUN')
    
    if dkey=='WSE':
        write_extract_mask(fp, ofp=ofp, maskType='binary', **kwargs)
    else:
        raise NotImplementedError(dkey)
    
    #===========================================================================
    # wrap
    #===========================================================================
    assert_type_fp(ofp, 'INUN_RLAY')
    
    return ofp


def write_wsh_clean(fp,
                    ofp=None, out_dir=None,
                    ):
    """filter a depths raster"""
    
    if ofp is None:
        ofp = _get_ofp(fp, out_dir, name='clean')
    
    mar_raw = load_array(fp, masked=True)
    
    ar_raw = mar_raw.data
    
    
    mar1 = ma.array(
                np.where(ar_raw>=0, ar_raw, 0.0), #filter
                 mask=mar_raw.mask, 
                 fill_value=mar_raw.fill_value)
    
    assert_wsh_ar(mar1)
    
    return write_array2(mar1, ofp, **get_profile(fp))
    



def write_rlay_to_poly(rlay_fp, 
                            dkey='WSE',
                            crs=None,
                            ofp=None, out_dir=None,
                            **kwargs):
    """
    
    Parame4ters
    --------
    dkey: str
    """
    #===========================================================================
    # defaults
    #===========================================================================
    if ofp is None:
        ofp = _get_ofp(rlay_fp, out_dir, name='inun', ext='geojson')
    
    if crs is None:
        crs = get_ds_attr(rlay_fp, 'crs')
        
    assert isinstance(crs, rio.crs.CRS)
    
    #===========================================================================
    # build the polygon
    #===========================================================================
    poly = rlay_to_poly(rlay_fp, dkey, **kwargs)
    
    assert isinstance(poly, sgeo.polygon.Polygon)
    
    #===========================================================================
    # #convert and write
    #===========================================================================
    gdf = gpd.GeoDataFrame(geometry=[poly], crs=crs)
    gdf.to_file(ofp)
        
    return ofp



    
def write_poly_to_rlay(poly_fp,
                       rlay_ref=None,
                       out_shape=None,
                       transform=None,
                       ofp=None, out_dir=None,
                       **kwargs):
    """write polygon to raster"""
    
    #===========================================================================
    # defaults
    #===========================================================================
    if ofp is None:
        ofp = _get_ofp(poly_fp, out_dir, name='inun', ext=os.path.splitext(rlay_ref)[1])
    
    #===========================================================================
    # get the mask
    #===========================================================================
    mask_ar = polyVlay_to_ar(poly_fp, 
                         rlay_ref=rlay_ref, out_shape=out_shape, transform=transform,
                             **kwargs)
    
    #===========================================================================
    # # Write the mask to the output raster
    #===========================================================================
    return write_array2(mask_ar, ofp, **get_profile(rlay_ref))

 
    
    
    
                       
        
#===============================================================================
# HIDDEN HELPERS---------
#===============================================================================
    
def _rlay_apply_hyd(rlay_fp, dkey, func, **kwargs):
    """special applier that recognizes our mask arrays"""
    
    if dkey in ['INUN_RLAY']:
        return rlay_mar_apply(rlay_fp, func, **kwargs)
    else:
        return rlay_ar_apply(rlay_fp, func, **kwargs)
        
    
def _get_hyd_ar(rlay_obj, dkey, **kwargs):
    """special array loader"""
    
    #allowing datasets here
    #assert_type_fp(rlay_obj, dkey)
    
    if dkey in ['INUN_RLAY']:
        return load_mask_array(rlay_obj, maskType='binary', **kwargs)
 
    else:
        return load_array(rlay_obj, masked=True, **kwargs)
        
def _get_inun_gdf(fp):
    gdf = gpd.read_file(fp)
    assert_inun_poly(gdf, msg=os.path.basename(fp))
    return gdf
        
 
#===============================================================================
# ASSERTIONS---------
#===============================================================================

def assert_type_fp(fp, dkey, msg=''):
    """check the file matches the dkey hydro expectations"""
    if not __debug__: # true if Python was not started with an -O option
        return 
    #__tracebackhide__ = True  
    
    #dkey check
    if not dkey in assert_func_d:
        raise AssertionError(f'unrecognized dkey {dkey}')
    
    #file type checking
    if not os.path.exists(fp):
        raise AssertionError(f'got bad filepath\n    {fp}\n'+msg)
    
    if dkey in ['WSH', 'WSE', 'DEM', 'INUN_RLAY']:
        assert is_raster_file(fp)
        _rlay_apply_hyd(fp, dkey, assert_func_d[dkey], msg=msg+f' w/ dkey={dkey}')
    elif dkey in ['INUN_POLY']:
        _get_inun_gdf(fp)
        
    else:
        raise KeyError(dkey)    

 
    
    
def assert_inun_ar(ar, msg=''):
    if not __debug__: # true if Python was not started with an -O option
        return 
    __tracebackhide__ = True
    
    assert_mask_ar(ar, msg=msg+' inun')
    if not ar.any():
        raise AssertionError(f'expect some Trues\n'+msg)
    
      

def assert_dem_ar(ar, msg=''):
    """check the array satisfies expectations for a DEM array"""
    if not __debug__: # true if Python was not started with an -O option
        return
    __tracebackhide__ = True 
    
    assert_masked_ar(ar, msg=msg)
    
    if not np.all(np.invert(ar.mask)):
        raise AssertionError(msg+': some masked values')
    
    
    
def assert_wse_ar(ar, msg=''):
    """check the array satisfies expectations for a WSE array"""
    if not __debug__: # true if Python was not started with an -O option
        return    
    __tracebackhide__ = True   
    
    assert_masked_ar(ar, msg=msg)    
    assert_partial_wet(ar.mask, msg=msg)
    
    
def assert_wsh_ar(ar, msg=''):
    """check the array satisfies expectations for a WD array"""
    if not __debug__: # true if Python was not started with an -O option
        return
    
    assert_masked_ar(ar, msg=msg)
    
    if not np.all(np.invert(ar.mask)):
        raise AssertionError(msg+': some masked values')
    
    if not np.min(ar)==0.0:
        raise AssertionError(msg+': expected zero minimum, got %.2f'%np.min(ar)) 
    
    if not np.max(ar)>0.0:
        raise AssertionError(msg+': zero maximum') 
    
    
    
def assert_partial_wet(ar, msg=''):
    """assert a boolean array has some trues and some falses (but not all)"""
    if not __debug__: # true if Python was not started with an -O option
        return
    #__tracebackhide__ = True 
    
    #assert isinstance(ar, ma.MaskedArray)
    assert 'bool' in ar.dtype.name
    
    if np.all(ar):
        raise AssertionError(msg+': all true')
    if np.all(np.invert(ar)):
        raise AssertionError(msg+': all false')
    
def assert_inun_poly(gdf, msg=''):
    assert len(gdf)==1, f'got {len(gdf)} feats\n'+msg
    assert gdf.iloc[0].geometry.geom_type in ['Polygon','MultiPolygon'], msg
    
    
    
    
assert_func_d = {
    'WSH':assert_wsh_ar,
    'WSE':assert_wse_ar,
    'DEM':assert_dem_ar,
    'INUN_RLAY':assert_inun_ar,
    'INUN_POLY':assert_inun_poly,
    }

#populate assertion functions by key
"""eaiser on the user"""
assert_func_d_fp = dict()
for k in assert_func_d.keys():
    assert_func_d_fp[k] = lambda fp, msg='':assert_type_fp(fp,k, msg=msg)
