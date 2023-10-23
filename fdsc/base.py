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
    assert_rlay_simple, _get_meta, assert_spatial_equal, get_ds_attr, write_array2
    )
from hp.riom import assert_masked_ar
from hp.hyd import assert_wse_ar, assert_dem_ar, assert_partial_wet, HydTypes
from hp.hyd import assert_wsh_ar as assert_wd_ar

 

 


class DscBaseWorker(object):
    """methods shared by all downscaler classes"""
    
    downscale=None
    nicknames_d={'CostGrow':'cgs', 
               'Basic':'rsmp',
               'SimpleFilter':'rsmpF', #basic + DEM filter
               'BufferGrowLoop':'bgl', 
               'Schumann14':'s14',

               }
    
    def __init__(self,
                 run_dsc_handle_d=None, 
                 **kwargs):
        """
        Parameters
        ----------
        run_dsc_handle_d: dict
            {methodName: callable function (takes kwargs)}
            
        """
        if run_dsc_handle_d is None: run_dsc_handle_d=dict()
        #=======================================================================
        # set caller funcs
        #=======================================================================
        miss_s = set(run_dsc_handle_d.keys()).difference(self.nicknames_d.keys())
        assert miss_s==set(), miss_s
 
        self.run_dsc_handle_d=run_dsc_handle_d
        
        #=======================================================================
        # init
        #=======================================================================
        super().__init__(**kwargs)
    
    
 

    def get_resolution_ratio(self, 
                             fp1,#coarse
                             fp2, #fine
                             ):
        s1 = get_ds_attr(fp1, 'res')[0]
        s2 = get_ds_attr(fp2, 'res')[0]
        return s1 / s2

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
            
            self.downscale = self.get_resolution_ratio(fp1, fp2)
            
        assert self.downscale>=1.0
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
 
    

#===============================================================================
# class DscBaseSession(DscBaseWorker, Session):
#     def __init__(self, 
#                  run_name='v1', #using v instead of r to avoid resolution confusion
#                  relative=True,
#                  **kwargs):
#  
#         super().__init__(run_name=run_name, relative=relative, **kwargs)
#===============================================================================
 
 
    

def rlay_extract(fp,
                 window=None, masked=True,
 
                 ):
    
    if not masked:
        raise NotImplementedError(masked)
    
    """load rlay data and arrays"""
    with rio.open(fp, mode='r') as ds:
        assert_rlay_simple(ds)
        stats_d = _get_meta(ds) 
 
        ar = ds.read(1, window=window, masked=masked)
        
        stats_d['null_cnt'] = ar.mask.sum()
        
    return stats_d, ar
    

def assert_dsc_res_lib(dsc_res_lib, level=1, msg=''):
    if not __debug__: # true if Python was not started with an -O option
        return
    assert isinstance(dsc_res_lib, dict)
    
    #===========================================================================
    # check keys
    #===========================================================================
    #check level 1 keys        
    #assert set(dsc_res_lib.keys()).difference(nicknames_d.keys())==set(['inputs'])
    
    for k0, d0 in dsc_res_lib.items():
        if level==1:
            assert set(d0.keys()).difference(['fp', 'meta', 'fp_rel'])==set(), f'{k0}: {d0.keys()}'
        elif level==2:
            for k1, d1 in d0.items():
                assert set(d1.keys()).difference(['fp', 'meta', 'fp_rel'])==set(), f'{k0}.{k1}: {d0.keys()}'
                
        else:
            raise KeyError(level)
                
        
def assert_type_fp(fp, dkey, **kwargs):
    return HydTypes(dkey).assert_fp(fp, **kwargs)
    
        
#===============================================================================
# def assert_dem_ar(ar, msg=''):
#     """check the array satisfies expectations for a DEM array"""
#     if not __debug__: # true if Python was not started with an -O option
#         return
#     
#     assert_masked_ar(ar, msg=msg)
#     
#     if not np.all(np.invert(ar.mask)):
#         raise AssertionError(msg+': some masked values')
#     
#     
#     
# def assert_wse_ar(ar, msg=''):
#     """check the array satisfies expectations for a WSE array"""
#     if not __debug__: # true if Python was not started with an -O option
#         return
#     
#     assert_masked_ar(ar, msg=msg)    
#     assert_partial_wet(ar.mask, msg=msg)
#     
#     
# def assert_wd_ar(ar, msg=''):
#     """check the array satisfies expectations for a WD array"""
#     if not __debug__: # true if Python was not started with an -O option
#         return
#     
#     assert_masked_ar(ar, msg=msg)
#     
#     if not np.all(np.invert(ar.mask)):
#         raise AssertionError(msg+': some masked values')
#     
#     if not np.min(ar)==0.0:
#         raise AssertionError(msg+': non-zero minimum') 
#     
#     if not np.max(ar)>0.0:
#         raise AssertionError(msg+': zero maximum') 
#     
#     
#     
# def assert_partial_wet(ar, msg=''):
#     """assert a boolean array has some trues and some falses (but not all)"""
#     if not __debug__: # true if Python was not started with an -O option
#         return
#     
#     #assert isinstance(ar, ma.MaskedArray)
#     assert 'bool' in ar.dtype.name
#     
#     if np.all(ar):
#         raise AssertionError(msg+': all true')
#     if np.all(np.invert(ar)):
#         raise AssertionError(msg+': all false')
#===============================================================================