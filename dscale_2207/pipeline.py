'''
Created on Mar. 28, 2023

@author: cefect

work pipelines for 2207_Dscale
'''
import os, pickle

import pandas as pd
from pandas import IndexSlice as idx



#===============================================================================
# IMPORTS------
#===============================================================================

from hp.pd import view
from hp.basic import dstr
from hp.rio import (get_bbox, write_clip, write_resample)

#from fdsc.control import Dsc_Session
from fdsc.eval.control import Dsc_Eval_Session
from fdsc.plot.control import Fdsc_Plot_Session

def load_pick(fp):
    with open(fp, 'rb') as f:
        d = pickle.load(f)
    assert isinstance(d, dict)
    print(f'loaded {len(d)} from \n    {fp}')
    return d



def run_downscale_and_eval(
        proj_lib,
        pick_lib=dict(),
 
        method_pars={
            'CostGrow': {}, 
            'Basic': {}, 
            'SimpleFilter': {}, 
            #'BufferGrowLoop': {}, 
            'Schumann14': dict(buffer_size=1.5, gridcells=True),
                         },
        vali_kwargs=dict(),
        **kwargs):
    """run all downscaleing methods then evaluate
    
    
    Pars
    ----------
    pick_lib, dict
        precompiled results
        
    """
 
    
    #init
    with Dsc_Eval_Session(**kwargs) as ses:
        
        def get_od(k):
            return os.path.join(ses.out_dir, k)
        
        #=======================================================================
        # clip
        #=======================================================================
        k='0clip'
        if not k in pick_lib:
            """clip using a rounded aoi on the WSE"""
            dem_fp, wse_fp = ses.p0_clip_rasters(proj_lib['dem1'], proj_lib['wse2'], out_dir=get_od(k))
            pick_lib[k] = ses._write_pick({'dem':dem_fp, 'wse':wse_fp}, resname=ses._get_resname(k))
        else:
            d = load_pick(pick_lib[k])
            dem_fp, wse_fp = d['dem'], d['wse']
        
        #=======================================================================
        # downscale
        #=======================================================================
 
        k = '1dsc'
        if not k in pick_lib:        
            dsc_res_lib = ses.run_dsc_multi(dem_fp, wse_fp, method_pars=method_pars,out_dir=get_od(k))
            
            pick_lib[k] = ses._write_pick(dsc_res_lib, resname=ses._get_resname(k))
            
        else:
            dsc_res_lib = load_pick(pick_lib[k])
            
        #=======================================================================
        # evalu
        #=======================================================================
        k = '2eval'
        if not k in pick_lib: 
            #extract downscaling filepaths
            fp_lib = ses._get_fps_from_dsc_lib(dsc_res_lib)
            
            #add rim simulations (hires)
            ses.bbox = get_bbox(wse_fp)
            fp_lib['RIM_hires'] = {'WSE1':ses.clip_rlay(proj_lib['wse1'])}
            
            #add rim (lowres)
            wse2_fp = ses.clip_rlay(proj_lib['wse2']) 
            fp_lib['RIM_lores'] = {'WSE1':write_resample(wse2_fp, ofp=get_od('wse2_clip_rsmp.tif'), 
                                          scale=ses.get_downscale(wse2_fp, dem_fp))}
            
            
            #extract validation data from project lib (re-key to match fperf.pipeline.run_vali_multi()
            d = {{'inun':'inun_fp', 'hwm':'hwm_pts_fp'}[k]:proj_lib[k] for k in ['inun', 'hwm']}
            vali_kwargs.update(d)
        
            #run validation
            dsc_vali_res_lib= ses.run_vali_multi_dsc(fp_lib, vali_kwargs=vali_kwargs,out_dir=get_od(k))
            
            #write pick
            pick_lib[k] = ses._write_pick(dsc_vali_res_lib, resname=ses._get_resname(k))
        else:
            dsc_vali_res_lib= load_pick(pick_lib[k])
        
    #===========================================================================
    # wrap
    #===========================================================================
    print(f'finished w/ pick_lib\n    {dstr(pick_lib)}')
 
    return dsc_vali_res_lib 
    
 
    

def run_plot(dsc_vali_res_lib, 
             dem_fp=None, inun_fp=None,
             init_kwargs=dict(),
             grids_mat_kg=dict(),
             hwm_scat_kg=dict(),
             inun_per_kg=dict(),
             ):
    """ plot downscaling and evalu results"""
    assert not 'aoi_fp' in  init_kwargs
    
    res_d = dict()
    
    with Fdsc_Plot_Session(**init_kwargs) as ses:
        log = ses.logger
        #extract filepaths
        serx = ses.load_run_serx(dsc_vali_res_lib = dsc_vali_res_lib)
        """
        dsc_vali_res_lib.keys()
        view(serx)
        serx.index.names
        serx.index.unique('analysis')
        serx['clip']
        print(dstr(dsc_vali_res_lib))
        """
 
        #fix some grid names
        serx = serx.rename({'wsh':'WSH', 'wse':'WSE', 'dem':'DEM'}, level=3)
 
        
        #=======================================================================
        # pull inputs from library
        #=======================================================================
        mdex = serx.index
        default_fp_d = serx['raw']['fp'][mdex.unique('simName')[0]].to_dict()
        if dem_fp is None:
            """not sure why this is 'clipped' but not on the clip level"""
            dem_fp = default_fp_d['DEM']
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
        for gridk in [
            #'WSH', #filters 'Basic'
            'WSE']:
            fp_d = serx['raw']['fp'].loc[idx[:, gridk]].to_dict()
            res_d[f'grids_mat_{gridk}'] = ses.plot_grids_mat(fp_d, gridk=gridk, 
                                         dem_fp=dem_fp,inun_fp=inun_fp, **grids_mat_kg)
 
        
        #=======================================================================
        # INUNDATION PERFORMANCe
        #======================================================================= 
        fp_df, metric_lib = ses.collect_inun_data(serx, 'WSH', raw_coln='raw')
        res_d['inun_perf'] = ses.plot_inun_perf_mat(fp_df.loc[:, ['WSH', 'CONFU']], 
                                                    metric_lib=metric_lib, **inun_per_kg)
        
        #=======================================================================
        # wrap
        #=======================================================================
        log.info(f'finished on\n{dstr(res_d)}')
        
    return res_d
            

    
 
        
        
    
             
    
    