'''
Created on Jan. 26, 2023

@author: cefect

numpy and rio have some unique handling of masked values.
Often, we have raster layers which are ONLY masks (i.e., no underlying data)
    and we want to load these as boolean arrays
    
this module contains functions for loading, writing, and converting

maskTypes: key for whether or not 0 values are natively masked
    binary: no native mask
    native: 0s are natively masked
    
0=null=True=masked=-9999
        
'''

import os
import numpy as np
 
import numpy.ma as ma
import rasterio as rio


from hp.rio import (
    rlay_apply, write_array2
    )


def load_mask_array(mask_fp, maskType='binary'):
    """helper to load mask from array and process
    
    
    here we assume a certain convention for storing mask rasters (in GDAL)
    then convert then to simple boolean arrays
    
    Parameters
    -----------
    maskType: str, default 'binary'
        how to treat the mask rlay

            
            """
    
    assert_f = lambda ar:assert_mask_ar_raw(ar, maskType=maskType)
    
    
    #===========================================================================
    # load by type
    #===========================================================================
    with rio.open(mask_fp, mode='r') as dataset:
        if maskType=='binary':
            mask_ar_raw = dataset.read(1,   masked=False)
            
            assert_f(mask_ar_raw)
                        
            mask_ar = np.where(mask_ar_raw==1, False, True)
            
        elif maskType=='native': 
            mask_ar_raw =  dataset.read(1,   masked=True)
            
            assert_f(mask_ar_raw)
            
            mask_ar = mask_ar_raw.mask
            
 
            
        else:
            raise KeyError(maskType)
        
    #===========================================================================
    # wrap
    #===========================================================================
    assert_mask_ar(mask_ar)
    
    return mask_ar


def write_array_mask(raw_ar, maskType='binary', 
                     ofp=None, out_dir=None,
                     nodata=-9999,
                     **kwargs):
    """write a boolean mask to a raster
    
    0=null=True=masked=-9999
    
    see load_mask_array"""
    
    assert_mask_ar(raw_ar)
    
    if maskType=='native':
        mask_raw_ar = ma.array(np.where(raw_ar, 0, 1),mask=raw_ar, fill_value=nodata)
    elif maskType=='binary':
        #no native mask.. just 1s and zeros
        mask_raw_ar = ma.array(np.where(raw_ar, 0, 1), mask=np.full(raw_ar.shape, False), fill_value=nodata)
    else:
        raise KeyError(maskType)
    
        
    if ofp is None:
        if out_dir is None:
            out_dir = os.path.expanduser('~')
        assert os.path.exists(out_dir)
                
        ofp = os.path.join(out_dir,'mask.tif')
    
    return write_array2(mask_raw_ar, ofp, **kwargs)


def write_extract_mask(raw_fp,  ofp=None, out_dir=None, maskType='binary', **kwargs):
 
    
    """extractc the native mask from a rlay as a separate raster. 0=masked"""
    
    #===========================================================================
    # retrieve
    #===========================================================================
    with rio.open(raw_fp, mode='r') as dataset:
        
        raw_ar = dataset.read(1,  masked=True)
        
        profile = dataset.profile
        
    #===========================================================================
    # filenames
    #===========================================================================
    ofp = _get_ofp(raw_fp, out_dir=out_dir, ofp=ofp)
        
    
    return write_array_mask(raw_ar.mask, ofp=ofp, maskType=maskType, **kwargs, **profile)
        
 
#===============================================================================
# helpers------
#===============================================================================
def _get_ofp(raw_fp, dkey='mask', out_dir=None, ofp=None):

    
    if ofp is None:
        if out_dir is None:
            out_dir = os.path.dirname(raw_fp)
        assert os.path.exists(out_dir)
    
        fname, ext = os.path.splitext(os.path.basename(raw_fp))                
        ofp = os.path.join(out_dir,f'{fname}_{dkey}{ext}')
        
    return ofp

#===============================================================================
# ASSERTIONS--------
#===============================================================================

def assert_mask(rlay,
               
                **kwargs):
    """check the passed rlay is a mask-like raster"""
    #assertion setup
    if not __debug__: # true if Python was not started with an -O option
        return
    __tracebackhide__ = True    

    
    #need to use the custom loader. this calls assert_mask_ar
    rlay_apply(load_mask_array, **kwargs)
    #rlay_ar_apply(rlay, assert_mask_ar, masked=False, **kwargs)
    

    
def assert_mask_ar_raw(ar,  maskType='binary'):
    """check raw raster array conforms to our mask speecifications
    
    usually we deal with processed masks... see assert_mask_ar
    
    see load_mask_array
    
    Parameters
    --------------
    maskType: str
        see load_mask_array
    """
    
    #===========================================================================
    # get test function based on mask typee
    #===========================================================================
    
    if maskType=='binary':
        assert not isinstance(ar, ma.MaskedArray)
        vals = set(np.unique(ar.ravel()))
        evals = {0.0, 1.0}        
        
    elif maskType=='native':
        assert isinstance(ar, ma.MaskedArray)
        """sets don't like the nulls apparently"""
        vals = set(np.unique(np.where(ar.mask, 1.0, ar.data).ravel()))
        evals = {1.0}  
    else:
        raise KeyError(maskType)
    
 
    assert vals.symmetric_difference(evals)==set(), f'got unexpected values for maskType {maskType}\n    {vals}'
    
def assert_mask_ar(ar):
    """check for processed array
    
    see load_mask_array
    """
    assert not isinstance(ar, ma.MaskedArray) 
    assert ar.dtype==np.dtype('bool')
