'''
Created on Mar. 28, 2023

@author: cefect

work pipelines for 2207_Dscale
'''
import os, pickle

import pandas as pd
from pandas import IndexSlice as idx

cm = 1 / 2.54

#===============================================================================
# IMPORTS------
#===============================================================================

from hp.pd import view
from hp.basic import dstr
from hp.rio import (get_bbox, write_clip, write_resample)

#from fdsc.control import Dsc_Session
from fdsc.eval.control import Dsc_Eval_Session
from fdsc.plot.control import Fdsc_Plot_Session



nickname_d2 = {
               'Hydrodyn. (s1)':'sim1',
               'Hydrodyn. (s2)':'sim2',
            }

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
        log = ses.logger
        ses.nicknames_d.update(nickname_d2)
        
        nd = {v:k for k,v in ses.nicknames_d.items()}
        
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
        log.info(f'finished {k} w/ {pick_lib[k]}\n\n')
        #=======================================================================
        # downscale
        #=======================================================================
 
        k = '1dsc'
        if not k in pick_lib:        
            dsc_res_lib = ses.run_dsc_multi(dem_fp, wse_fp, method_pars=method_pars,out_dir=get_od(k))
            
            pick_lib[k] = ses._write_pick(dsc_res_lib, resname=ses._get_resname(k))
            
        else:
            dsc_res_lib = load_pick(pick_lib[k])
        log.info(f'finished {k} w/ {pick_lib[k]}\n\n') 
        #=======================================================================
        # eval-----------
        #=======================================================================
        k = '2eval'
        if not k in pick_lib: 
            #extract downscaling filepaths
            fp_lib = ses._get_fps_from_dsc_lib(dsc_res_lib)
            
            #add rim simulations (hires)
            ses.bbox = get_bbox(wse_fp)
            fp_lib[nd['sim1']] = {'WSE1':ses.clip_rlay(proj_lib['wse1'])}
            
            #add rim (lowres)
            wse2_fp = ses.clip_rlay(proj_lib['wse2']) 
            fp_lib[nd['sim2']] = {'WSE1':write_resample(wse2_fp, ofp=get_od('wse2_clip_rsmp.tif'), 
                                          scale=ses.get_downscale(wse2_fp, dem_fp))}
            
            
            #extract validation data from project lib (re-key to match fperf.pipeline.run_vali_multi()
            d = {{'inun':'inun_fp', 'hwm':'hwm_pts_fp'}[k]:proj_lib[k] for k in ['inun', 'hwm']}
            vali_kwargs.update(d)
        
            #run validation
            dsc_vali_res_lib= ses.run_vali_multi_dsc(fp_lib,out_dir=get_od(k), **vali_kwargs)
            
            #write pick
            pick_lib[k] = ses._write_pick(dsc_vali_res_lib, resname=ses._get_resname(k))
        else:
            dsc_vali_res_lib= load_pick(pick_lib[k])
        
    #===========================================================================
    # wrap
    #===========================================================================
    print(f'finished w/ pick_lib\n    {dstr(pick_lib)}')
 
    return dsc_vali_res_lib 
    
 

def run_downscale_and_eval_multiRes(
        proj_lib,dsc_l,
        pick_lib=dict(),
 
        method_pars={
            'CostGrow': {}, 
            'Basic': {}, 
            'SimpleFilter': {}, 
            #'BufferGrowLoop': {}, 
            #'Schumann14': dict(buffer_size=1.5, gridcells=True),
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
        log = ses.logger
        ses.nicknames_d.update(nickname_d2)
        
        nd = {v:k for k,v in ses.nicknames_d.items()}
        
        def get_od(k):
            return os.path.join(ses.out_dir, k)
        print(proj_lib.keys())
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
        log.info(f'finished {k} w/ {pick_lib[k]}\n\n')
        
        #=======================================================================
        # Agg DEMs
        #=======================================================================
        k='1dems'
        if not k in pick_lib: 
            #build aggregated DEMs
            downscale_base = ses.get_resolution_ratio(wse_fp, dem_fp)
            dem_scale_l = [downscale_base] + [downscale_base/e for e in dsc_l] #change to be relative to the DEM1
            
            
            dem_fp_d = ses.build_agg_dem(dem_fp, dem_scale_l, logger=log, out_dir=get_od(k))
            
            pick_lib[k] = ses._write_pick(dem_fp_d, resname=ses._get_resname(k))
            
        else:
            dem_fp_d = load_pick(pick_lib[k])
        log.info(f'finished {k} w/ {pick_lib[k]}\n\n') 
        
        #=======================================================================
        # downscale
        #=======================================================================
 
        k = '2dsc'
        if not k in pick_lib:        
            dsc_res_lib = ses.run_dsc_multi_mRes(wse_fp, dem_fp_d, method_pars=method_pars,out_dir=get_od(k))
            
            #add some upper level meta
            """
            print(dstr(dsc_res_lib['inputs']['inputs1']))
            """
            dsc_res_lib['inputs']['inputs1']['meta']['dsc_l']=dsc_l.copy()
            dsc_res_lib['inputs']['inputs1']['fp']['DEM1']=dem_fp
            
            #add inputs
            
            #write pick
            pick_lib[k] = ses._write_pick(dsc_res_lib, resname=ses._get_resname(k))
            
        else:
            dsc_res_lib = load_pick(pick_lib[k])
        log.info(f'finished {k} w/ {pick_lib[k]}\n\n')
        
        #=======================================================================
        # depth grids 
        #=======================================================================
        k='3wsh'
        if not k in pick_lib:
            #extract the filepaths        
            fp_lib = ses._get_fps_from_dsc_lib(dsc_res_lib, level=2)
            
            """
            print(dstr(fp_lib))
            """
            
            #build WSH
            wsh_lib = ses.build_wsh(fp_lib, out_dir=get_od(k))
            
            #add back to the results container
            dsc_res_lib2 = ses._add_fps_to_dsc_lib(dsc_res_lib, wsh_lib, 'WSH1')

            #write
            pick_lib[k] = ses._write_pick(dsc_res_lib2, resname=ses._get_resname(k))
            
        else:
            dsc_res_lib2 = load_pick(pick_lib[k])
            
        log.info(f'finished {k} w/ {pick_lib[k]}\n\n')
        
        #=======================================================================
        # volume stats-----------
        #=======================================================================
        
        k='4stats'
        if not k in pick_lib:  
            
            #extract the filepaths        
            fp_lib = ses._get_fps_from_dsc_lib(dsc_res_lib2, level=2)
            """
            print(dstr(fp_lib))
            """
            
            #compute stats
            stats_lib = ses.run_stats_multiRes(fp_lib, out_dir=get_od(k))
            
            #update
            dsc_res_lib3 = dsc_res_lib2.copy()
            for k0, d0 in stats_lib.items():
                for k1, d1 in d0.items():
                    dsc_res_lib3[k0][k1]['grid_stats'] = d1.copy()

 
            #write
            pick_lib[k] = ses._write_pick(dsc_res_lib3, resname=ses._get_resname(k))
            
        else:
            dsc_res_lib3 = load_pick(pick_lib[k])
        log.info(f'finished {k} w/ {pick_lib[k]}\n\n') 
        
 
        
    #===========================================================================
    # wrap
    #===========================================================================
    print(f'finished w/ pick_lib\n    {dstr(pick_lib)}')
 
    return dsc_res_lib3 
    

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
        ses.nicknames_d.update(nickname_d2)
                
        log = ses.logger
        nd = {v:k for k,v in ses.nicknames_d.items()}
        
        #extract filepaths
        serx = ses.load_run_serx(dsc_vali_res_lib = dsc_vali_res_lib)
        """
        dsc_vali_res_lib.keys()
        view(serx)
        serx.index.names
        serx.index.unique('simName')
        serx['clip']
        print(dstr(dsc_vali_res_lib))
        """
 
        #fix some grid names
        serx = serx.rename({'wsh':'WSH', 'wse':'WSE', 'dem':'DEM'}, level=3)
 
        #set display order
        sim_order_l1 = ['sim2', 'rsmp', 'rsmpF', 'cgs', 's14', 'sim1']
        sim_order_l2 = [nd[k] for k in sim_order_l1] #fancy names
        
        serx = serx.loc[idx[:, :, sim_order_l2, :]]
        #=======================================================================
        # pull inputs from library
        #=======================================================================
        mdex = serx.index
        default_fp_d = serx['raw']['fp'][mdex.unique('simName')[0]].to_dict()
        if dem_fp is None:
            """not sure why this is 'clipped' but not on the clip level"""
            dem_fp = default_fp_d['DEM']
            
        if inun_fp is None:
            inun_fp = default_fp_d['INUN_POLY']
            
        
        #=======================================================================
        # HWM performance (all)
        #=======================================================================
        hwm_gdf = ses.collect_HWM_data(serx['hwm']['fp'],write=False)
       
        hwm_scat_kg.update(dict(metaKeys_l = ['rvalue','rmse']))
        res_d['hwm_scat'] = ses.plot_HWM_scatter(hwm_gdf, **hwm_scat_kg)
       
        #=======================================================================
        # grid plots
        #=======================================================================
        for gridk in [
            #'WSH', #filters 'Basic'
            'WSE']:

            
            res_d[f'grids_mat_{gridk}'] = ses.plot_grids_mat_fdsc(serx, gridk, dem_fp, inun_fp,
                                                                   grids_mat_kg=grids_mat_kg)
 
 
        #=======================================================================
        # INUNDATION PERFORMANCe
        #=======================================================================
        #prep data 
        fp_df, metric_lib = ses.collect_inun_data(serx, 'WSH', raw_coln='raw')
        
        #reorder
        ml = ['criticalSuccessIndex','hitRate', 'falseAlarms', 'errorBias']        
        for k, d in metric_lib.copy().items():
            metric_lib[k] = {k:d[k] for k in ml}
        
                
        sim_order_l2.remove('Hydrodyn. (s2)') #remove this... plot basic instead
        dfi = fp_df.loc[sim_order_l2, ['WSH', 'CONFU']] 
        
        #plot
        res_d['inun_perf'] = ses.plot_inun_perf_mat(dfi,metric_lib=metric_lib, 
                                rowLabels_d={nd['rsmp']:'Basic/Hydrodyn. (s2)'},
                                #arrow_kwargs_lib={'flow1':dict(xy_loc = (0.66, 0.55))},
 
                                **inun_per_kg)
        
        #=======================================================================
        # wrap
        #=======================================================================
        log.info(f'finished on\n{dstr(res_d)}')
        
    return res_d
            

def run_plot_multires(dsc_res_lib, 
                      sim_color_d = {'cgs': '#e41a1c', 'rsmp': '#ff7f00', 'rsmpF': '#999999'},
 
             init_kwargs=dict(),
 
             ):
    """ plot downscaling and grid stat results for multi-Res"""
    assert not 'aoi_fp' in  init_kwargs
    
    res_d = dict()
    
    from fdsc.plot.multiRes import Fdsc_MultiRes_Plot_Session as Session
    
    with Session(**init_kwargs) as ses:
        log = ses.logger
        #=======================================================================
        # data setup
        #=======================================================================
        ses.nicknames_d.update(nickname_d2)
                
        
        nd = {v:k for k,v in ses.nicknames_d.items()}
        
        #extract filepaths
        serx = ses.load_run_serx_multiRes_stats(dsc_res_lib)
 
        #fix some grid names
        lvl_d = {lvlName:i for i, lvlName in enumerate(serx.index.names)}
        serx = serx.rename({'WSH1':'WSH'}, level=lvl_d['dataType'])
 
        #set display order
        sim_order_l1 = ['sim2', 'rsmp', 'rsmpF', 'cgs', 's14', 'sim1']
        sim_order_l2 = [k for k in sim_order_l1 if k in serx.index.unique('simName')]
        
        
        #get resolution keys
        
        
        #=======================================================================
        # plot stats  
        #=======================================================================
        serx1 = serx.loc[idx[:, :, 'WSH',:]].loc[idx[sim_order_l2,:,['vol']]].reorder_levels(
            ['varName', 'simName', 'dscale']) #row, col, xval
        
        serx1 = serx.loc[idx[:,:,'WSH','vol']].loc[idx[sim_order_l2, :]].reorder_levels(
            ['simName', 'dscale']) #color, xval, yval
        
        #reshape into simple df
        df = serx1.to_frame().unstack()        
        df.columns = df.columns.droplevel(0)
 
        #secondary axis
        dsc_res_d = serx.loc[idx['rsmp', :, 'meta', 'resolution']].to_dict()
 
        ses.plot_multiRes_stats_single(df, color_d=sim_color_d, base_scale=dsc_res_d[1.0],
                                       subplots_kwargs=dict(figsize=(19*cm, 10*cm)))
                                       
                                       
        
        
        
    
             
    
    