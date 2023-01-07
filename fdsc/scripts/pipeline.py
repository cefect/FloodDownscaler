'''
Created on Jan. 7, 2023

@author: cefect

integrated downscaling and validation
'''

import os
from fdsc.scripts.dsc import Dsc_Session
from fdsc.scripts.valid import ValidateSession
from definitions import wrk_dir, src_name
from hp.basic import today_str

class PipeSession(Dsc_Session, ValidateSession):
    pass

def run_dsc_vali(
        wse2_rlay_fp,
        dem1_rlay_fp,
        aoi_fp=None,
        
        run_name='zone1_cds_v1',
        out_dir=None,
        **kwargs
        ):
    """generate downscale then compute metrics (one option)"""
    
    #===========================================================================
    # defaults
    #===========================================================================
    with PipeSession(**kwargs) as ses:
        pass
        
 
        
 
    
    
        run_downscale(wse2_rlay_fp, dem1_rlay_fp, out_dir=out_dir, **dsc_kwargs)