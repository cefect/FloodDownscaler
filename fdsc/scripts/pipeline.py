'''
Created on Jan. 7, 2023

@author: cefect

integrated downscaling and validation
'''

import os, datetime
from fdsc.scripts.dsc import Dsc_Session
from fdsc.scripts.valid import ValidateSession
from definitions import wrk_dir, src_name
from hp.basic import today_str

def now():
    return datetime.datetime.now()

class PipeSession(Dsc_Session, ValidateSession):
    pass

def run_dsc_vali(
        wse2_rlay_fp,
        dem1_rlay_fp,
        wse1V_fp=None,
        dsc_kwargs=dict(dryPartial_method = 'costDistanceSimple'),
 
        **kwargs
        ):
    """generate downscale then compute metrics (one option)
    
    Parameters
    ------------
    wse1_V_fp: str
        filepath to wse1 raster (for validation)
    """
    
    #===========================================================================
    # defaults
    #===========================================================================
    with PipeSession(**kwargs) as ses:
        start = now()
        log = ses.logger.getChild('r')
        #=======================================================================
        # downscale
        #=======================================================================
        wse1_dp_fp = ses.run_dsc(wse2_rlay_fp,dem1_rlay_fp,**dsc_kwargs)
        assert os.path.exists(wse1_dp_fp)
        #=======================================================================
        # validate
        #=======================================================================
        _ = ses.run_vali(true_fp=wse1V_fp, pred_fp=wse1_dp_fp)
        
        #=======================================================================
        # wrap
        #=======================================================================
        log.info(f'finished in {now()-start}')
 