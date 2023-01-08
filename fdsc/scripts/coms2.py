'''
Created on Jan. 6, 2023

@author: cefect

shared by all sessions
'''
import pandas as pd
import numpy as np
import numpy.ma as ma

from hp.oop import Session

class Master_Session(Session):
    def __init__(self, 
                 run_name='v1', #using v instead of r to avoid resolution confusion
                 **kwargs):
 
        super().__init__(run_name=run_name, **kwargs)
        
    def _write_meta(self, meta_lib, **kwargs):
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('meta', subdir=False,ext='.xls',  **kwargs)
        
        #write dict of dicts to frame
        with pd.ExcelWriter(ofp, engine='xlsxwriter') as writer:
            for tabnm, d in meta_lib.items():
                pd.Series(d).to_frame().to_excel(writer, sheet_name=tabnm, index=True, header=True)
        
        log.info(f'wrote meta (w/ {len(meta_lib)}) to \n    {ofp}')
        
        return ofp
    
def assert_masked_ar(ar, msg=''):
    """check the array satisfies expectations for a masked array"""
    if not __debug__: # true if Python was not started with an -O option
        return
    
    if not isinstance(ar, ma.MaskedArray):
        raise AssertionError(msg+' bad type ' + type(ar))
    if not 'float' in ar.dtype.name:
        raise AssertionError(msg+' bad dtype ' + ar.dtype.name)
    
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