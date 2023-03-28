'''
Created on Mar. 28, 2023

@author: cefect

work pipeline for 2207_Dscale
'''
import os, pickle


from fdsc.control import Dsc_Session
from fdsc.eval.control import  Dsc_Eval_Session

def run_downscale(
        proj_lib,
        init_kwargs=dict(),
        method_pars={'CostGrow': {}, 
                         'Basic': {}, 
                         'SimpleFilter': {}, 
                         'BufferGrowLoop': {}, 
                         'Schumann14': {},
                         },
        **kwargs):
    """run all downscaleing methods"""
    

            
    if 'aoi' in init_kwargs:
        init_kwargs['aoi_fp'] = init_kwargs.pop('aoi')
    
    #init
    with Dsc_Session(**init_kwargs) as ses:
        
        dem_fp, wse_fp = ses.p0_clip_rasters(proj_lib['dem1'], proj_lib['wse2'])
        
        dsc_res_lib = ses.run_dsc_multi(dem_fp, wse_fp, method_pars=method_pars, 
                                 copy_inputs=False,
                                 **kwargs)
        
        logger = ses.logger
        
    return dsc_res_lib, logger
    
    
def run_eval(dsc_res_lib,
                     init_kwargs=dict(),
                     vali_kwargs=dict(),
                     **kwargs):
    """ run evaluation on downscaling results"""
    assert not 'aoi_fp' in  init_kwargs
    
    with Dsc_Eval_Session(**init_kwargs) as ses:
 
        #extract filepaths
        fp_lib = ses._get_fps_from_dsc_lib(dsc_res_lib)
        
        #run validation
        return ses.run_vali_multi_dsc(fp_lib, vali_kwargs=vali_kwargs, **kwargs)
        
        
    
             
    
    