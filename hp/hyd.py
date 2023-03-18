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
def get_depth(dem_fp, wse_fp, out_dir = None, ofp=None):
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
    
    #logic checks
    assert not dem_ar.mask.any(), f'got {dem_ar.mask.sum()} masked values in dem array \n    {dem_fp}'
    assert wse_ar.mask.any()
    assert not wse_ar.mask.all()
    
    #===========================================================================
    # calc
    #===========================================================================
    #simple subtraction
    wd1_ar = wse_ar - dem_ar
    
    #identify dry
    dry_bx = np.logical_or(
        wse_ar.mask, wse_ar.data<dem_ar.data
        )
    
    assert not dry_bx.all().all()
    
    #rebuild
    wd2_ar = np.where(~dry_bx, wd1_ar.data, 0.0)
    
    
    #check we have no positive depths on the wse mask
    assert not np.logical_and(wse_ar.mask, wd2_ar>0.0).any()
    
    #===========================================================================
    # write
    #===========================================================================
    
    #convert to masked
    wd2M_ar = ma.array(wd2_ar, mask=np.isnan(wd2_ar), fill_value=wse_ar.fill_value)
    
    assert not wd2M_ar.mask.any(), 'depth grids should have no mask'
    
    return write_array2(wd2M_ar, ofp, masked=False, **get_profile(wse_fp))

def get_wse(dem_fp, wd_fp, out_dir = None, ofp=None):
    """add dem and wse to get a depth grid"""
    
    assert_spatial_equal(dem_fp, wse_fp)
    
    if ofp is None:
        if out_dir is None:
            out_dir = tempfile.gettempdir()
        if not os.path.exists(out_dir):os.makedirs(out_dir)
        
        fname = os.path.splitext( os.path.basename(wse_fp))[0] + '_wse.tif'
        ofp = os.path.join(out_dir,fname)
    
    #===========================================================================
    # load
    #===========================================================================
    dem_ar = load_array(dem_fp, masked=True)
    
    wd_ar = load_array(wd_fp, masked=True)
    
    #logic checks
    assert not dem_ar.mask.any(), f'got {dem_ar.mask.sum()} masked values in dem array \n    {dem_fp}'
    assert wse_ar.mask.any()
    assert not wse_ar.mask.all()
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
    
    
def assert_wd_ar(ar, msg=''):
    """check the array satisfies expectations for a WD array"""
    if not __debug__: # true if Python was not started with an -O option
        return
    
    assert_masked_ar(ar, msg=msg)
    
    if not np.all(np.invert(ar.mask)):
        raise AssertionError(msg+': some masked values')
    
    if not np.min(ar)==0.0:
        raise AssertionError(msg+': non-zero minimum') 
    
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