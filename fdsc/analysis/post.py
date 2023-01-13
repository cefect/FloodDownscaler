'''
Created on Jan. 9, 2023

@author: cefect

data analysis on multiple downscale results
'''


import logging, os, copy, datetime, pickle, pprint
import numpy as np
import pandas as pd
import rasterio as rio

from rasterio.plot import show

import matplotlib.pyplot as plt
import matplotlib

from hp.plot import Plotr, get_dict_str


from fdsc.analysis.valid import ValidateSession

def dstr(d):
    return pprint.pformat(d, width=30, indent=0.3, compact=True, sort_dicts =False)

class Plot_rlays_wrkr(object):
    
    def collect_rlay_fps(self, run_lib, **kwargs):
        """collect the filepaths from the run_lib"""
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('load_metas', **kwargs)
        
        fp_lib={k:dict() for k in run_lib.keys()}
        metric_lib = {k:dict() for k in run_lib.keys()}
        
        #=======================================================================
        # pull for each
        #=======================================================================
        for k0, d0 in run_lib.items(): #simulation name
            for k1, d1 in d0.items(): #cat0
                for k2, d2 in d1.items():
                    if k1=='smry':
                        if k2 in [
                            #'wse1', 'wse2', 
                            'dep2']:
                            fp_lib[k0][k2]=d2                    
                        
                    elif k1=='vali':
                        if k2=='inun':
                            for k3, v3 in d2.items():
                                if k3=='confuGrid_fp':
                                    fp_lib[k0][k3]=v3
                                else:
                                    metric_lib[k0][k3]=v3
                        elif k2=='grid':
                            for k3, v3 in d2.items():
                                if k3=='dep1':
                                    fp_lib[k0][k3]=v3
                                elif k3=='true_dep_fp':
                                    dep1V = v3
                    else:
                        pass
             
        #=======================================================================
        # get validation
        #=======================================================================
        fp_lib = {**{'vali':{'dep1':dep1V}}, **fp_lib} #order matters
 
 
        log.info('got fp_lib:\n%s\n\nmetric_lib:\n%s'%(dstr(fp_lib), dstr(metric_lib)))
        
 
        return fp_lib, metric_lib
        
    
    def plot_rlay_mat(self,
                      fp_lib, metric_lib=None, 
                      figsize=(9,5),
            **kwargs):
        """matrix plot comparing methods for downscaling: rasters
        
        rows: 
            valid, methods
        columns
            depthRaster r2, depthRaster r1, confusionRaster
        """
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('load_metas',ext='.png', **kwargs)
        
        log.info(f'on {list(fp_lib.keys())}')
        
        #=======================================================================
        # setup figure
        #=======================================================================
        row_keys = list(fp_lib.keys())
        col_keys = list(fp_lib[list(fp_lib.keys())[-1]].keys())
        
 
        fig, ax_d = self.get_matrix_fig(row_keys, col_keys, logger=log, 
                                        set_ax_title=False, figsize=figsize,
                                        constrained_layout=True,
                                        )
        
        
        #=======================================================================
        # colormap
        #=======================================================================
        raise NotImplementedError('stopped herer')
        #build a custom color map        
        cmap = matplotlib.colors.ListedColormap(cvals)
        
        #discrete normalization
        norm = matplotlib.colors.BoundaryNorm(
                                            np.array([0]+ckeys)+1, #bounds that capture the data 
                                              ncolors=len(ckeys),
                                              #cmap.N, 
                                              extend='neither',
                                              clip=True,
                                              )
        
        #=======================================================================
        # plot loop
        #=======================================================================
        for rowk, d0 in ax_d.items():
            for colk, ax in d0.items():
                
                
                if not colk in fp_lib[rowk]:
                    continue 
                
                fp=fp_lib[rowk][colk]
                log.info(f'plotting {rowk}x{colk}: {os.path.basename(fp)}')
                with rio.open(fp, mode='r') as ds:
                    ar_raw = ds.read(1, window=None, masked=True)
                    
                    #mask zeros in depths
                    if 'dep' in colk:
                        ar = np.where(ar_raw==0, np.nan, ar_raw)
                    else:
                        ar = ar_raw.data
                     
                    #raster plot
                    ax_img = show(ar, 
                                  transform=ds.transform, 
                                  ax=ax,contour=False, cmap='viridis', interpolation='nearest')
                    
                    #hide labels
                    ax.get_xaxis().set_ticks([])
                    ax.get_yaxis().set_ticks([])
                    
                    #add text
                    if colk=='confuGrid_fp' and isinstance(metric_lib, dict):
                        ax.text(0.1, 0.9, get_dict_str(metric_lib[rowk]), 
                                transform=ax.transAxes, va='top', fontsize=8, color='black')
                        
                    
        #=======================================================================
        # turn off useless axis
        #=======================================================================
        for ax in (ax_d['vali']['dep2'], ax_d['vali']['confuGrid_fp']):
            #===================================================================
            # ax.cla()
            # ax.spines['top'].set_visible(False)
            # ax.spines['right'].set_visible(False)
            # ax.spines['left'].set_visible(False)
            #===================================================================
            ax.set_axis_off()
        
                    
        #=======================================================================
        # wrap
        #=======================================================================
        log.info('finished')
        return self.output_fig(fig, ofp=ofp, logger=log)
        """
        plt.show()
        """
                    
                
                
                
                
        
        
        
class Plot_samples_wrkr(object):
    def plot_sample_mat(self, 
                        **kwargs):
        """matrix plot comparing methods for downscaling: sampled values
        
        rows: methods
        columns:
            depth histogram, difference histogram, correlation plot
            
        same as Figure 5 on RICorDE paper"""

class PostSession(Plot_rlays_wrkr, Plotr, ValidateSession):
    "Session for analysis on multiple downscale results and their validation metrics"
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
        
        print(pprint.pformat(run_lib['nodp'], width=30, indent=0.3, compact=True, sort_dicts =False))
        
        self.run_lib = copy.deepcopy(run_lib)
        
        #get teh summary d
        
        smry_d = {k:v['smry'] for k,v in run_lib.items()}
        return run_lib, smry_d    
        
        

    def _load_pick_lib(self, fp_d, log):
        res_d = dict()
        data_j = None
        for k, fp in fp_d.items():
            log.debug(f'for {k} loading {fp}')
            assert os.path.exists(fp)
            with open(fp, 'rb') as f:
                data = pickle.load(f)
                assert isinstance(data, dict)
        #consistency check
                if not data_j is None:
                    assert set(data_j.keys()).symmetric_difference(data.keys()) == set()
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
    
    #===========================================================================
    # def get_rlay_fps(self, run_lib=None, **kwargs):
    #     """get the results rasters from the run_lib"""
    #     log, tmp_dir, out_dir, ofp, resname = self._func_setup('get_rlay_fps', **kwargs) 
    #     if run_lib is None: run_lib=self.run_lib
    #     
    #     log.info(f'on {run_lib.keys()}')
    # 
    #     run_lib['nodp']['smry'].keys()
    #     
    #===========================================================================
        
    

            
        
        
        
        
 