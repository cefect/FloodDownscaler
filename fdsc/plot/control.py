'''
Created on Mar. 28, 2023

@author: cefect

plotting downsample performance results
'''


import logging, os, copy, datetime, pickle, pprint, math
import numpy as np
import pandas as pd
from pandas import IndexSlice as idx

 
 

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

from hp.oop import Session
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
from fdsc.base import DscBaseWorker
from fperf.plot.pipeline import PostSession



class Fdsc_Plot_Session(Fdsc_Plot_Base, DscBaseWorker, PostSession):
    "plotting downsample performance"
    
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
    
    def plot_grids_mat_fdsc(self, serx, gridk,dem_fp, inun_fp,
                            grids_mat_kg=dict(),
                            **kwargs):
        """wrapper for plot_grids_mat to add custom arrows"""
        
        #=======================================================================
        # setup
        #=======================================================================
        log, tmp_dir, out_dir, ofp, resname = self._func_setup(
            f'grids{gridk.upper()}', ext='.'+self.output_format, **kwargs)
        
        #=======================================================================
        # data prep
        #=======================================================================
        fp_d = serx['raw']['fp'].loc[idx[:, gridk]].to_dict()
        
        #=======================================================================
        # plot
        #=======================================================================
        ax_d, fig = self.plot_grids_mat(fp_d, gridk=gridk, 
                                     dem_fp=dem_fp,inun_fp=inun_fp, 
                                     #arrow_kwargs_lib={'flow1':dict(xy_loc = (0.66, 0.55))},
                                     **grids_mat_kg)
        
        #=======================================================================
        # add custom arrows
        #=======================================================================
        def add_annot(ax, txt, xy_loc):
            ax.annotate(txt,
                    xy=xy_loc,
                    xycoords='axes fraction',
                    #xytext=(xy_loc[0]-0.25, xy_loc[1]),
                    textcoords='axes fraction',
                    #arrowprops=dict(facecolor='black', shrink=0.08, alpha=0.7),
                    bbox=dict(boxstyle="circle,pad=0.3", fc="black", lw=0.0,alpha=0.9 ),
                    color='white',
                    )
            
            log.debug(f'added {txt} at {str(xy_loc)}')
            
 
        #wet-partials (Resample)
        add_annot(ax_d['r0']['c2'], 'A', (0.2, 0.35))
        
        #dry-partials (TerrainFilter)
        add_annot(ax_d['r1']['c0'], 'B', (0.75, 0.7))
        
        #Cost Grow limits
        add_annot(ax_d['r0']['c1'], 'C', (0.75, 0.85))
 
        #Schuymann isolated
        add_annot(ax_d['r1']['c1'], 'D', (0.56, 0.89))
        
        #flow on RR tracks
        add_annot(ax_d['r1']['c2'], 'E', (0.25, 0.81))
        

 
        return self.output_fig(fig, ofp=ofp, logger=log, dpi=600)
    


        
                
        
            
         