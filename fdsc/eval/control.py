'''
Created on Mar. 27, 2023

@author: cefect

running evaluation on downscaling results
'''

from fdsc.base import DscBaseSession
from fperf.pipeline import ValidateSession

class Dsc_Eval_Session(DscBaseSession, ValidateSession):
    
    def run_dsc_vali_multi(self,
                           **kwargs):
        """build validation on downsample results"""
        
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('rdvX',  **kwargs)
        
 