'''
Created on Mar. 19, 2023

@author: cefect
'''
#===============================================================================
# IMPORTS------
#===============================================================================
import datetime, os, pickle
import pandas as pd
 

from hp.oop import Session

from hp.tests.tools.rasters import get_rand_ar

#===============================================================================
# CLASSES--------
#===============================================================================


class BaseWorker(object):
    pass

class BaseSession(BaseWorker, Session):
    def __init__(self, 
                 run_name='v1', #using v instead of r to avoid resolution confusion
                 relative=True,
                 **kwargs):
 
        super().__init__(run_name=run_name, relative=relative, **kwargs)
        
    def _write_meta(self, meta_lib, **kwargs):
        """write a dict of dicts to a spreadsheet"""
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('meta', subdir=False,ext='.xls',  **kwargs)
        
        #write dict of dicts to frame
        with pd.ExcelWriter(ofp, engine='xlsxwriter') as writer:
            for tabnm, d in meta_lib.items():
                pd.Series(d).to_frame().to_excel(writer, sheet_name=tabnm, index=True, header=True)
        
        log.info(f'wrote meta (w/ {len(meta_lib)}) to \n    {ofp}')
        
        return ofp
    
    def _write_pick(self, data, **kwargs):
        """dump data to a pickle"""
        
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('w', subdir=False,ext='.pkl',  **kwargs)
        
        with open(ofp, 'wb') as handle:
            pickle.dump(data, handle)
            
        log.info(f'wrote {type(data)} to \n    {ofp}')
        return ofp
    

