'''
Created on Jan. 6, 2023

@author: cefect

shared by all sessions
'''
import pandas as pd
from hp.oop import Session

class Master_Session(Session):
    def __init__(self, 
                 run_name='v1', #using v instead of r to avoid resolution confusion
                 **kwargs):
 
        super().__init__(run_name=run_name, **kwargs)
        
    def _write_meta(self, meta_lib, **kwargs):
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('meta', subdir=False,ext='.xls',  **kwargs)
        
 
        with pd.ExcelWriter(ofp, engine='xlsxwriter') as writer:
            for tabnm, d in meta_lib.items():
                pd.Series(d).to_frame().to_excel(writer, sheet_name=tabnm, index=True, header=True)
        
        log.info(f'wrote meta (w/ {len(meta_lib)}) to \n    {ofp}')
        
        return ofp