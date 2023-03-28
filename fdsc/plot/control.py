'''
Created on Mar. 28, 2023

@author: cefect
'''


import logging, os, copy, datetime, pickle, pprint, math
import numpy as np
import pandas as pd
import rasterio as rio
import geopandas as gpd
import scipy

from rasterio.plot import show

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.image import AxesImage
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

#earthpy
#===============================================================================
# #import earthpy as et
# import earthpy.spatial as es
# import earthpy.plot as ep
#===============================================================================

from hp.plot import get_dict_str, hide_text
from hp.pd import view
from hp.rio import (    
    get_ds_attr,get_meta, SpatialBBOXWrkr
    )
from hp.rio_plot import RioPlotr
from hp.err_calc import get_confusion_cat, ErrorCalcs
from hp.gpd import get_samples
from hp.fiona import get_bbox_and_crs


from fdsc.plot.base import Fdsc_Plot_Base


class Fdsc_Plot_Session(Fdsc_Plot_Base):
    "Session for analysis on multiple downscale results and their validation metrics"
    
    #see self._build_color_d(mod_keys)

    
    
    
    def __init__(self, 
                 run_name = None,
                 **kwargs):
 
        if run_name is None:
            run_name = 'post_v1'
        super().__init__(run_name=run_name, **kwargs)
        
    
    def load_metas(self, fp_d, 
                        **kwargs):
        """load metadata from a collection of downscale runs"""
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('load_metas', **kwargs) 
        
        run_lib = self._load_pick_lib(fp_d, log)
        
        log.info(f'loaded for {len(run_lib)} runs\n    {run_lib.keys()}')
        
        #print contents
        k1=list(run_lib.keys())[0]
        print(f'\nfor {k1}')
        #print(pprint.pformat(run_lib[k1], width=30, indent=0.3, compact=True, sort_dicts =False))
        
        self.run_lib = copy.deepcopy(run_lib)
        
        #=======================================================================
        # #get teh summary d
        #=======================================================================
        
        smry_d = {k:v['smry'] for k,v in run_lib.items()}
        #print(f'\nSUMMARY:')
        #print(pprint.pformat(smry_d, width=30, indent=0.3, compact=True, sort_dicts =False))
        #=======================================================================
        # wrap
        #=======================================================================
        
        return run_lib, smry_d    
        
        

    def _load_pick_lib(self, fp_d, log):
        res_d = dict()
        data_j = None
        for k, fp in fp_d.items():
            log.debug(f'for {k} loading {fp}')
            assert os.path.exists(fp), f'bad filepath for \'{k}\':\n    {fp}'
            with open(fp, 'rb') as f:
                data = pickle.load(f)
                assert isinstance(data, dict)
                
                """not forcing this any more (hydro validations are keyed differently)
                #consistency check
                if not data_j is None:
                    assert set(data_j.keys()).symmetric_difference(data.keys()) == set()"""
                #store
                #res_d[k] = pd.DataFrame.from_dict(data)
                res_d[k] = data
                data_j = data
        
        return res_d

    def load_metric_set(self, fp_d,
                 **kwargs):
        """load a set of pipeline valiMetrics.pkl
        
        
        collect these from meta_lib.pkl?"""
        #=======================================================================
        # defaults
        #=======================================================================
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('load_metric_set', **kwargs) 
        
        assert isinstance(fp_d, dict)
        
        #=======================================================================
        # loop and load dictionaries
        #=======================================================================
        res_d = self._load_pick_lib(fp_d, log)
                
        for k,v in res_d.items():
            assert isinstance(v, pd.Series), f'bad type on \'{k}\':{type(v)}'
        #=======================================================================
        # concat                
        #=======================================================================
        dx = pd.concat(res_d, axis=1, names=['run', 'metricSet'])
        
        assert dx.notna().all().all()
        #=======================================================================
        # wrap
        #=======================================================================
        log.info(f'loaded {str(dx.shape)}')
        
        return dx
    
    def collect_runtimes(self, run_lib, **kwargs):
        """log runtimes for each"""
        
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('runtimes', **kwargs) 
        
        res_lib={k:dict() for k in run_lib.keys()} 
        for k0, d0 in run_lib.items():
            try:
                res_lib[k0] = d0['dsc']['smry']['tdelta']
            except:
                del res_lib[k0]
                log.warning(f'no key on {k0}...skipping')
                
        #=======================================================================
        # convert to minutes
        #=======================================================================
        convert_d = {k:v/1 for k,v in res_lib.items()}
        dstr = pprint.pformat(convert_d, width=30, indent=0.3, compact=True, sort_dicts =False)
        
        log.info(f'collected runtimes for {len(res_lib)} (in seconds): \n{dstr}')
        
        return res_lib
                
        
            
        
        
        
def basic_post_pipeline(meta_fp_d, 
                      sample_dx_fp=None,
                      hwm_pick_fp=None,
                      
                      ses_init_kwargs = dict(),
                      inun_perf_kwargs= dict(),
                      samples_mat_kwargs=dict(),
                      hwm3_kwargs=dict(),
                      hyd_hwm_kwargs=dict(),
                      rlay_res_kwargs=dict()
                      ):
    """main runner for generating plots"""    
    
    res_d = dict()
    with PostSession(**ses_init_kwargs) as ses:
        
        #load the metadata from teh run
        run_lib, smry_d = ses.load_metas(meta_fp_d)
        
        #ses.collect_runtimes(run_lib)
        
        #=======================================================================
        # HWM performance (all)
        #=======================================================================
        #=======================================================================
        # fp_lib, metric_lib = ses.collect_HWM_data(run_lib)
        # gdf = ses.concat_HWMs(fp_lib,pick_fp=hwm_pick_fp)
        # res_d['HWM3'] = ses.plot_HWM_3x3(gdf, metric_lib=metric_lib, **hwm3_kwargs)
        #=======================================================================
 
        #=======================================================================
        # hydrodyn HWM performance
        #=======================================================================
  #=============================================================================
  #       ses.collect_hyd_fps(run_lib) 
  #          
  #       ses.plot_hyd_hwm(gdf.drop('geometry', axis=1), **hyd_hwm_kwargs)
  # 
  #=============================================================================
        
        #=======================================================================
        # RASTER PLOTS
        #=======================================================================
        #get rlays
        rlay_fp_lib, metric_lib = ses.collect_rlay_fps(run_lib)        
        res_d['rlay_res'] = ses.plot_rlay_res_mat(rlay_fp_lib, metric_lib=metric_lib, **rlay_res_kwargs)
        
 
        #=======================================================================
        # INUNDATION PERFORMANCe
        #======================================================================= 
        #res_d['inun_perf'] = ses.plot_inun_perf_mat(rlay_fp_lib, metric_lib, **inun_perf_kwargs)
         
 
 

 
        #=======================================================================
        # sample metrics
        #=======================================================================
#===============================================================================
#         ses.logger.info('\n\nSAMPLES\n\n')
#         try: 
#             del run_lib['nodp'] #clear this
#         except: pass
#         
#         df, metric_lib = ses.collect_samples_data(run_lib, sample_dx_fp=sample_dx_fp)
#         
#         """switched to plotting all trues per simulation"""
#  
#         df_wet=df
# 
#         res_d['ssampl_mat'] =ses.plot_samples_mat(df_wet, metric_lib, **samples_mat_kwargs)
#===============================================================================
        
    print('finished on \n    ' + pprint.pformat(res_d, width=30, indent=True, compact=True, sort_dicts =False))
    return res_d
 
