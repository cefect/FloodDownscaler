"""

plotting pipeline
"""

#===============================================================================
# IMPORTS-----------
#===============================================================================

import logging, os, copy, datetime, pickle, pprint, math
import numpy as np
import pandas as pd
from pandas import IndexSlice as idx
import rasterio as rio
import geopandas as gpd
import scipy

from rasterio.plot import show

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.image import AxesImage
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

#earthpy
 
from hp.basic import dstr
from hp.plot import Plotr, get_dict_str, hide_text, cm
from hp.pd import view, nested_dict_to_dx, dict_to_multiindex
from hp.rio import (    
    get_ds_attr,get_meta, SpatialBBOXWrkr
    )
from hp.err_calc import get_confusion_cat, ErrorCalcs
from hp.gpd import get_samples
from hp.fiona import get_bbox_and_crs


from fperf.pipeline import ValidateSession
 
from fperf.plot.inun import Plot_inun_peformance
from fperf.plot.hwm import Plot_samples_wrkr, Plot_HWMS
from fperf.plot.grids import Plot_grids

class PostSession(Plot_inun_peformance, Plot_samples_wrkr, Plot_HWMS,Plot_grids,
                   ValidateSession):
    "Session for analysis on multiple downscale results and their validation metrics"
    
    #see self._build_color_d(mod_keys)
    
    """not used... was also on fdsc.plot.base.Fdsc_Plot_Base   (makes more sense there
    #sim_color_d = {'CostGrow': '#e41a1c', 'Basic': '#377eb8', 'SimpleFilter': '#984ea3','Schumann14': '#ffff33', 'WSE2': '#f781bf', 'WSE1': '#999999'}
    """
                   
                   
        
        
 
    def __init__(self, 
                 run_name = None,
                 **kwargs):
 
        if run_name is None:
            run_name = 'plot1'
            
        super().__init__(run_name=run_name, **kwargs)
        
        
        
        
        
    

    def _get_run_serx(self, run_lib2):
        serx = pd.DataFrame(dict_to_multiindex(run_lib2), index=['val']).iloc[0, :]
        #fix the names
        serx.index.set_names(['analysis', 'dataType', 'simName', 'varName'], inplace=True)
        
        return serx
    


    def load_run_serx(self,  pick_fp=None, dsc_vali_res_lib=None,
                      relative=None, base_dir=None,
                      unix_base_dir=None,
                        **kwargs):
        """load output pickle from run_vali_multi() and convert into a serx
        
        pars
        -------
        
        
        returns
        ---------------
        serx
            analysis type
                data type
                    simulation name
                    
        """
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('load_run_serx', **kwargs)
        if relative is None: relative=self.relative
        if base_dir is None: base_dir=self.base_dir 
        #=======================================================================
        # load pick
        #=======================================================================
        if dsc_vali_res_lib is None:
            with open(pick_fp, 'rb') as f:
                run_lib = pickle.load(f)
                assert isinstance(run_lib, dict)            
     
            log.info(f'loaded for {len(run_lib)} runs from \'{os.path.basename(pick_fp)}\'\n    {list(run_lib.keys())}')
        else:
            run_lib = dsc_vali_res_lib
            
        #print contents
        #=======================================================================
        # k1=list(run_lib.keys())[0]
        # print(f'\nfor {k1}')
        # print(pprint.pformat(run_lib[k1], width=30, indent=0.3, compact=True, sort_dicts =False))
        #=======================================================================
 
        #=======================================================================
        # move lvl0 (sim name) to lvl2
        #=======================================================================
        """at this stage, easeir to work with metric, fp, and meta libs separately"""
        
        
        run_lib2 = dict()
        for k0, v0 in run_lib.items(): #sim name
            for k1, v1 in v0.items(): #analysis type (hwm, inun)
                #add teh page
                if not k1 in run_lib2:
                    run_lib2[k1] = dict()
                    
                for k2, v2 in v1.items(): #data type (metric, fp, meta)
                    
                    #add the page
                    if not k2 in run_lib2[k1]:
                        run_lib2[k1][k2] = dict()
                        
                    #add the entryu
                    run_lib2[k1][k2][k0]=v2
                    #print(k1, k2, k0)
        
        #=======================================================================
        # convert from relative
        #=======================================================================
        if relative:
            assert os.path.exists(base_dir)
            for k0, d0 in run_lib2.items(): #analysis type
                
                fp_rel_lib = d0.pop('fp_rel')             
                fp_lib = {k:dict() for k in fp_rel_lib.keys()}
                
                for k2, fp_rel_d in fp_rel_lib.items(): #simName
                    
                    for k3, fp_rel in fp_rel_d.items():
                        
                        if k3 in ['hwm_pts_fp', 'wd_fp', 'pred_inun_fp']:
                            """these are inputs... need a nicer way to organise""" 
                            continue
                        
                        """looks like relative pathing didnt work..."""
                        #/home/bryant/LS/10_IO/2207_dscale/outs/aoi07t_0414/r01
                        if not unix_base_dir is None:
                            assert unix_base_dir in fp_rel, 'got bad unix_base_dir'
                            #unix_base_dir = str(base_dir).replace('\\', '/').replace('l:', '/home/bryant/LS')+'/'
                            fp_rel_fix = fp_rel.replace(unix_base_dir+'/', '').replace('/', '\\')
                        else:
                            fp_rel_fix=fp_rel
                    
                        #print(k2, fp_rel_d)
                        fp = os.path.join(base_dir, fp_rel_fix)
                        assert os.path.exists(fp), f'{k0}.{k2}.{k3}\n    {fp}'
                        
                        fp_lib[k2][k3]=fp
                
                #update
                run_lib2[k0]['fp'] = fp_lib
                
        else:
            #remove fp_rel
 
            for k0, d0 in run_lib2.items(): #analysis type
                if 'fp_rel' in d0:
                    del d0['fp_rel']
            
 
        #print(dstr(run_lib2))
 
        #print(run_lib2.keys())    
        #=======================================================================
        # convert multindex
        #=======================================================================
        for k0, d0 in run_lib2.items():
            if k0 in ['clip', 'raw']:
                if 'meta' in d0:
                    d0.pop('meta') #not sure why these don't work
            #===================================================================
            # for k1, d1 in d0.items():
            #     print(f'{k0}.{k1}: {d1.keys()}\n\n')
            #     for k2, d2 in d1.items():
            #         print(f'    {k2}:{d2.keys()}')
            #===================================================================
                
        
        serx = self._get_run_serx(run_lib2)
        """
        view(serx)
        """
        
        #print each level name and unique values 
        #=======================================================================
        # for name, lvl_vals in zip(serx.index.names, serx.index.levels):
        #     print(name, lvl_vals.values.tolist())
        #=======================================================================
        
 
        #=======================================================================
        # wrap
        #=======================================================================
        assert set(serx.index.unique('analysis')).difference(
            ['hwm', 'inun', 'raw', 'clip', 'inun_wsh2', 'clip_wsh2', 'inputs'])==set()
            
        dif = set(['hwm', 'inun', 'raw']).difference(serx.index.unique('analysis'))
        if not dif==set():
            log.warning(f'missing some expected analysis categories {dif}')
            
            
        assert set(serx.index.unique('dataType')).symmetric_difference(
            ['metric', 'fp', 'meta'])==set(), serx.index.unique('dataType')
        
        return serx 
    
    
    def collect_var(self, serx,
                    pick_fp=None,write=False,
                         varName='pred_inun_fp',
                         **kwargs):
        """intelligent collection of a dictionary of variable names"""
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('collect_var', **kwargs)
        
 
        #=======================================================================
        # build--------
        #=======================================================================
        if pick_fp is None:
            #=======================================================================
            # loop and load each prediction
            #=======================================================================
            res_d = serx.loc[idx[:, varName]].to_dict()
 
            #===================================================================
            # write it
            #===================================================================
            if write:
                self._write_pick(res_d, resname=varName.replace('_fp', ''), out_dir=tmp_dir)
 
            
        #=======================================================================
        # compiled-------
        #=======================================================================
        else:
            assert not write
            log.warning(f'loading from \n    {pick_fp}')
            with open(pick_fp, 'rb') as f:
                res_d = pickle.load(f)
                
        #=======================================================================
        # wrap
        #=======================================================================
        assert isinstance(res_d, dict)
        log.info(f'loaded \'{varName}\' w/ {len(res_d)}\n    {res_d}')
        
        return res_d
        
 
    def collect_inun_data(self, serx, gridk, inun_coln='inun', raw_coln='clip',):
        """collect all the inundation data"""
        
        #frame of filepahts
        fp_df = pd.concat({
                'CONFU':serx[inun_coln]['fp'].loc[idx[:, 'confuGrid_fp']],
                gridk:serx[raw_coln]['fp'].loc[idx[:, gridk]]
                },axis=1).dropna(axis=0, how='any')
                
        #drop wse2
        """unlike HWMs, inundation performance between basic and wse2 is identical"""
        fp_df=fp_df.drop('Hydrodyn. (s2)', axis=0, errors='ignore')
        
        #extract the metrics
        s1 = serx.loc[idx[inun_coln, 'metric', :]]
        metric_lib = {}
        for (k1, k2), val in s1.items():
            if k1 not in metric_lib:
                metric_lib[k1] = {}
            metric_lib[k1][k2] = val
            
        #check
        assert set(fp_df.index).difference(metric_lib.keys())==set()
        
        return fp_df, metric_lib
    """
    view(serx['clip']['fp'])
    """

