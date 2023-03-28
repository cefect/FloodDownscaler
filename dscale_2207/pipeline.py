'''
Created on Mar. 28, 2023

@author: cefect

work pipeline for 2207_Dscale
'''

from fdsc.control import Dsc_Session

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
    
    #setup init
    for k in ['proj_name', 'aoi', 'crs']:
        if k in proj_lib:
            init_kwargs[k]=proj_lib[k]
            
    if 'aoi' in init_kwargs:
        init_kwargs['aoi_fp'] = init_kwargs.pop('aoi')
    
    #init
    with Dsc_Session(**init_kwargs) as ses:
        
        dem_fp, wse_fp = ses.p0_clip_rasters(proj_lib['dem1'], proj_lib['wse2'])
        
        return ses.run_dsc_multi(dem_fp, wse_fp, method_pars=method_pars, **kwargs)
    
    