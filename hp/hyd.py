'''
Created on Mar. 18, 2023

@author: cefect

special hydraulic helpers
'''

import datetime, os, tempfile
import pandas as pd
import numpy as np
import numpy.ma as ma
 

from hp.rio import (
    assert_rlay_simple, get_stats, assert_spatial_equal, get_ds_attr, write_array2, assert_masked_ar,
    load_array, get_profile
    )

#===============================================================================
# RASTERS -------
#===============================================================================
def get_wsh_rlay(dem_fp, wse_fp, out_dir = None, ofp=None):
    """add dem and wse to get a depth grid"""
    
    assert_spatial_equal(dem_fp, wse_fp)
    
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
        
    

def get_wse_rlay(dem_fp, wd_fp, out_dir = None, ofp=None):
    """add dem and wse to get a depth grid"""
    
    assert_spatial_equal(dem_fp, wd_fp)
    
    if ofp is None:
        if out_dir is None:
            out_dir = tempfile.gettempdir()
        if not os.path.exists(out_dir):os.makedirs(out_dir)
        
        fname = os.path.splitext( os.path.basename(wd_fp))[0] + '_wse.tif'
        ofp = os.path.join(out_dir,fname)
    
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
    
    
    
 
#===============================================================================
# ASSERTIONS---------
#===============================================================================
def assert_dem_ar(ar, msg=''):
    """check the array satisfies expectations for a DEM array"""
    if not __debug__: # true if Python was not started with an -O option
        return
    
    assert_masked_ar(ar, msg=msg)
    
    if not np.all(np.invert(ar.mask)):
        raise AssertionError(msg+': some masked values')
    
    
    
def assert_wse_ar(ar, msg=''):
    """check the array satisfies expectations for a WSE array"""
    if not __debug__: # true if Python was not started with an -O option
        return
    
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
    
    #assert isinstance(ar, ma.MaskedArray)
    assert 'bool' in ar.dtype.name
    
    if np.all(ar):
        raise AssertionError(msg+': all true')
    if np.all(np.invert(ar)):
        raise AssertionError(msg+': all false')