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



def _dataset_to_mar(dataset, maskType, window):
    """return a boolean array based on the mask type"""
    assert dataset.nodata == -9999
    #=======================================================================
    # raster has 0s and 1s
    #=======================================================================
    if maskType == 'binary':
        mask_ar_raw = dataset.read(1, masked=False, window=window)
        #check the loaded array conforms to the mask expectations
        assert_mask_ar_raw(mask_ar_raw, maskType=maskType)
        mask_ar = np.where(mask_ar_raw == 1, False, True)
    elif maskType == 'native':
        mask_ar_raw = dataset.read(1, masked=True, window=window)
        #check the loaded array conforms to the mask expectations
        assert_mask_ar_raw(mask_ar_raw, maskType=maskType)
        mask_ar = mask_ar_raw.mask
    else:
        raise KeyError(maskType)
 
    return mask_ar

def load_mask_array(mask_obj, maskType='binary', window=None,):
    """helper to load boolean array from rlay mask and process
    
    
    here we assume a certain convention for storing mask rasters (in GDAL)
    then convert then to simple boolean arrays
    
    Parameters
    -----------
    maskType: str, default 'binary'
        how to treat the mask rlay
            binary: 0=True=wet

            
    """
 
    #===========================================================================
    # load by type
    #===========================================================================
    if isinstance(mask_obj, str):        
        with rio.open(mask_obj, mode='r') as dataset:
            mask_ar = _dataset_to_mar(dataset, maskType, window)
    
    else:
        mask_ar = _dataset_to_mar(mask_obj, maskType, window)
        
    #===========================================================================
    # wrap
    #===========================================================================
    assert_mask_ar(mask_ar)
    
    return mask_ar


def write_array_mask(raw_ar, 
                     maskType='binary',
                     ofp=None, out_dir=None,
                     nodata=-9999,
                     bbox=None,
                     transform=None, #dummy identify
                     **kwargs):
    """write a boolean mask to a raster. 
    
    name is a bit misleading:
        0=null=True=masked=-9999
    
    see load_mask_array"""
    
    #===========================================================================
    # defaults
    #===========================================================================
    if ofp is None:
        if out_dir is None:
            out_dir = os.path.expanduser('~')
        assert os.path.exists(out_dir)
                
        ofp = os.path.join(out_dir,'mask.tif')
    
    #===========================================================================
    # load the array
    #===========================================================================
    assert_mask_ar(raw_ar, msg='expects a boolean array')
    
    if maskType=='native':
        mask_raw_ar = ma.array(np.where(raw_ar, 0, 1),mask=raw_ar, fill_value=nodata)
    elif maskType=='binary':
        #no native mask.. just 1s and zeros
        mask_raw_ar = ma.array(np.where(raw_ar, 0, 1), mask=np.full(raw_ar.shape, False), fill_value=nodata)
    else:
        raise KeyError(maskType)
    
    #===========================================================================
    # get spatial parameters
    #===========================================================================


    
    return write_array2(mask_raw_ar, ofp, nodata=nodata,transform=transform, bbox=bbox, **kwargs)


def write_extract_mask(raw_fp,  
                       ofp=None, out_dir=None, 
                       maskType='binary',invert=False,
                       window=None, bbox=None, 
                       **kwargs):
 
    
    """extractc the mask from a rlay as a separate raster. 0=masked"""
    ofp = _get_ofp(raw_fp, out_dir=out_dir, ofp=ofp)
    #===========================================================================
    # retrieve
    #===========================================================================
    with rio.open(raw_fp, mode='r') as dataset:
        
        if window is None and (not bbox is None):
            window = rio.windows.from_bounds(*bbox.bounds, transform=dataset.transform)
            transform = rio.windows.transform(window, dataset.transform)
        else:
            transform=dataset.transform
            assert bbox is None
 
        
        raw_ar = dataset.read(1,  masked=True, window=window)
        assert_masked_ar(raw_ar)
        assert np.any(raw_ar.mask), 'passed raster has no mask'
        
        ds_prof = dataset.profile
        
        #assert not np.any(np.isnan(np.unique(raw_ar.filled()))), f'unmasked nulls on \n    {raw_fp}'
        
        
        
    #overwrite the nodata
    ds_prof['nodata'] = -9999
    ds_prof['transform'] = transform
    #===========================================================================
    # manipulate mask
    #===========================================================================
    if invert:
        mask = np.invert(raw_ar.mask)
    else:
        mask = raw_ar.mask
        
    #===========================================================================
    # filenames
    #===========================================================================
    
        
    
    return write_array_mask(mask, ofp=ofp, maskType=maskType, **kwargs, **ds_prof)
        
def rlay_mar_apply(rlay_fp, func, maskType='binary', **kwargs):
    """apply a function to a mask on an rlay
    
    similar to hp.rio.rlay_ar_apply (but less flexible)
    """
    mask_ar = load_mask_array(rlay_fp,  maskType=maskType)
    
    return func(mask_ar, **kwargs)
        

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

def assert_mask_fp(rlay_fp,
               msg='',
                **kwargs):
    """check the passed rlay is a mask-like raster"""
    #assertion setup
    if not __debug__: # true if Python was not started with an -O option
        return
    __tracebackhide__ = True    

    assert isinstance(rlay_fp, str)
    assert os.path.exists(rlay_fp)
    
    #need to use the custom loader. this calls assert_mask_ar
    try:
        load_mask_array(rlay_fp, **kwargs)
    except Exception as e:
        raise TypeError(f'{e}\n    not a mask: '+msg)
    #rlay_ar_apply(rlay, assert_mask_ar, masked=False, **kwargs)
    

    
def assert_mask_ar_raw(ar,  maskType='binary', msg=''):
    """check raw raster array conforms to our mask speecifications
    
    usually we deal with processed masks... see assert_mask_ar
    
    see load_mask_array
    
    Parameters
    --------------
    maskType: str
        see load_mask_array
    """
    if not __debug__: # true if Python was not started with an -O option
        return 
    __tracebackhide__ = True
    
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
        
        #raise NotImplementedError('not sure about this one....')
        evals = {1.0}  
    else:
        raise KeyError(maskType)
    
 
    if not vals.symmetric_difference(evals)==set():
        raise AssertionError(f'got unexpected values for maskType {maskType}\n    {vals}')
    

def assert_masked_ar(ar, msg=''):
    """check the array satisfies expectations for a masked array
        not to be comfused with a MASK array
     
    NOTE: to call this on a raster filepath, wrap with rlay_ar_apply:
        rlay_ar_apply(wse1_dp_fp, assert_wse_ar, msg='result WSe')
    """
    if not __debug__: # true if Python was not started with an -O option
        return
     
    if not isinstance(ar, ma.MaskedArray):
        raise AssertionError(msg+'\n     bad type ' + str(type(ar)))
    if not 'float' in ar.dtype.name:
        raise AssertionError(msg+'\n     bad dtype ' + ar.dtype.name)
     
    #check there are no nulls on the data
    if np.any(np.isnan(ar.filled())):
        raise AssertionError(msg+f'\n    got {np.isnan(ar.data).sum()}/{ar.size} nulls outside of mask')
    
    if np.all(ar.mask):
        raise AssertionError(msg+f'\n    passed array is fully m asked')
         
  
    
def assert_mask_ar(ar, msg=''):
    """check if mask array
        not to be confused with MASKED array
   
    TODO: accept masks (we should be able to keep the mask information)
    
    see load_mask_array
    """
    if not __debug__: # true if Python was not started with an -O option
        return 
    __tracebackhide__ = True
    
    assert not isinstance(ar, ma.MaskedArray), msg
    assert ar.dtype==np.dtype('bool'), msg
