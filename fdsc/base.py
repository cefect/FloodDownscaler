'''
Created on Jan. 6, 2023

@author: cefect

base objects and functions for all dsc scripts
'''
import datetime, os
import pandas as pd
import numpy as np
import numpy.ma as ma
import rasterio as rio

from hp.oop import Session
from hp.rio import (
    assert_rlay_simple, get_stats, assert_spatial_equal, get_ds_attr, write_array2, assert_masked_ar
    )

nicknames_d = {'costGrow':'cgs', 
               'basicBilinear':'none',
               'simpleFilter':'nodp', 
               'bufferGrowLoop':'bgl', 
               'schumann14':'s14'}

def now():
    return datetime.datetime.now()


class Dsc_basic(object):
    
    downscale=None

 
    def get_downscale(self, fp1, fp2, **kwargs):
        """compute the scale difference between two layers
        
        Parameters
        --------
        fp1: str
            low-res (coarse)
        fp2: str
            hi-res (fine)
        """
        
        if self.downscale is None:
            s1 = get_ds_attr(fp1, 'res')[0]
            s2 = get_ds_attr(fp2, 'res')[0]
            
            assert s1 > s2
            
            self.downscale = s1 / s2
            
        
        return self.downscale
    
    def get_wse_dem_filter(self, wse_fp, dem_fp,  **kwargs):
        """filter the passed WSe by the DEM"""
        #=======================================================================
        # defaults
        #=======================================================================
        meta_d= dict()
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('wdFilter', **kwargs)
        
        assert_spatial_equal(wse_fp, dem_fp)
        
        #=======================================================================
        # compute
        #=======================================================================
        with rio.open(dem_fp, mode='r') as dem_ds:
            dem1_ar = dem_ds.read(1, window=None, masked=True)
            assert_dem_ar(dem1_ar)
            meta_d['s1_size'] = dem1_ar.size
            
            
            with rio.open(wse_fp, mode='r') as wse1_ds:
                wse1_ar = wse1_ds.read(1, window=None, masked=True)
                assert_wse_ar(wse1_ar)
                meta_d['pre_dem_filter_mask_cnt'] = wse1_ar.mask.sum().sum()
                
                # extend mask to include violators mask
                wse_wp_bx = np.logical_or(
                    wse1_ar.mask, 
                    wse1_ar.data <= dem1_ar.data)
                
                # build new array
                wse1_ar2 = ma.array(wse1_ar.data, mask=wse_wp_bx)
                assert_wse_ar(wse1_ar2)
                
                #wrap
                meta_d['post_dem_filter_mask_cnt'] = wse1_ar2.mask.sum().sum()
                delta_cnt = meta_d['post_dem_filter_mask_cnt'] - meta_d['pre_dem_filter_mask_cnt']
                log.info(f'filtered {delta_cnt} of {dem1_ar.size} additional cells w/ DEM')
                assert delta_cnt >= 0, 'dem filter should extend the mask'
                prof = wse1_ds.profile
                
        #=======================================================================
        # wrap
        #======================================================================= 
        log.debug(f'finsihed on {ofp}')
        return write_array2(wse1_ar2, ofp, **prof), meta_d
 
    

class Master_Session(Session):
    def __init__(self, 
                 run_name='v1', #using v instead of r to avoid resolution confusion
                 **kwargs):
 
        super().__init__(run_name=run_name, **kwargs)
        
    def _write_meta(self, meta_lib, **kwargs):
        """write a dict of dicts to a spreadsheet"""
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('meta', subdir=False,ext='.xls',  **kwargs)
        
        #write dict of dicts to frame
        with pd.ExcelWriter(ofp, engine='xlsxwriter') as writer:
            for tabnm, d in meta_lib.items():
                pd.Series(d).to_frame().to_excel(writer, sheet_name=tabnm, index=True, header=True)
        
        log.info(f'wrote meta (w/ {len(meta_lib)}) to \n    {ofp}')
        
        return ofp
    

def rlay_extract(fp,
                 window=None, masked=True,
 
                 ):
    
    if not masked:
        raise NotImplementedError(masked)
    
    """load rlay data and arrays"""
    with rio.open(fp, mode='r') as ds:
        assert_rlay_simple(ds)
        stats_d = get_stats(ds) 
 
        ar = ds.read(1, window=window, masked=masked)
        
        stats_d['null_cnt'] = ar.mask.sum()
        
    return stats_d, ar
    

    
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