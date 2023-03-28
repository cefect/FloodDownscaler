'''
Created on Mar. 28, 2023

@author: cefect

work pipeline for 2207_Dscale
'''
import os, pickle

import pandas as pd
from pandas import IndexSlice as idx



#===============================================================================
# IMPORTS------
#===============================================================================

from hp.pd import view
from hp.basic import dstr

from fdsc.control import Dsc_Session
from fdsc.eval.control import  Dsc_Eval_Session
from fdsc.plot.control import Fdsc_Plot_Session



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
        dsc_vali_res_lib= ses.run_vali_multi_dsc(fp_lib, vali_kwargs=vali_kwargs, **kwargs)
        logger=ses.logger
    
    return dsc_vali_res_lib, logger
    
    

def run_plot(dsc_vali_res_lib, 
             dem_fp=None, inun_fp=None,
             init_kwargs=dict(),
             grids_mat_kg=dict(),
             hwm_scat_kg=dict(),
             inun_per_kg=dict(),
             ):
    """ run evaluation on downscaling results"""
    assert not 'aoi_fp' in  init_kwargs
    
    res_d = dict()
    
    with Fdsc_Plot_Session(**init_kwargs) as ses:
        log = ses.logger
        #extract filepaths
        serx = ses.load_run_serx(dsc_vali_res_lib = dsc_vali_res_lib)
        """
        view(serx)
        serx.index.names
        serx.index.unique('analysis')
        serx['clip']
        """
        
        
        #=======================================================================
        # pull inputs from library
        #=======================================================================
        mdex = serx.index
        default_fp_d = serx['raw']['fp'][mdex.unique('simName')[0]].to_dict()
        if dem_fp is None:
            """not sure why this is 'clipped' but not on the clip level"""
            dem_fp = default_fp_d['dem']
        if inun_fp is None:
            inun_fp = default_fp_d['inun']
            
        
        #=======================================================================
        # HWM performance (all)
        #=======================================================================
        hwm_gdf = ses.collect_HWM_data(serx['hwm']['fp'],write=False)
        res_d['hwm_scat'] = ses.plot_HWM_scatter(hwm_gdf, **hwm_scat_kg)
        #=======================================================================
        # grid plots
        #=======================================================================
        for gridk in ['wsh', 'wse']:
            fp_d = serx['raw']['fp'].loc[idx[:, gridk]].to_dict()
            res_d[f'grids_mat_{gridk}'] = ses.plot_grids_mat(fp_d, gridk=gridk, 
                                         dem_fp=dem_fp,inun_fp=inun_fp, **grids_mat_kg)
            
        
        #=======================================================================
        # INUNDATION PERFORMANCe
        #======================================================================= 
        fp_df, metric_lib = ses.collect_inun_data(serx, gridk, raw_coln='raw')
        res_d['inun_perf'] = ses.plot_inun_perf_mat(fp_df, metric_lib=metric_lib, **inun_per_kg)
        
        #=======================================================================
        # wrap
        #=======================================================================
        log.info(f'finished on\n{dstr(res_d)}')
        
    return res_d
            

    
 
        
        
    
             
    
    