'''
Created on Jan. 9, 2023

@author: cefect

data analysis on multiple downscale results
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


 
from fdsc.analysis.valid.v_ses import ValidateSession
from fdsc.base import nicknames_d

#nicknames_d['Hydrodynamic (s1)']='WSE1'
cm = 1/2.54
 
#nicknames_d2 = {v:k for k,v in nicknames_d.items()}





def dstr(d):
    return pprint.pformat(d, width=30, indent=0.3, compact=True, sort_dicts =False)

 
class PostBase(RioPlotr):
    """base worker"""
    
    sim_color_d = {'CostGrow': '#e41a1c', 'Basic': '#377eb8', 'SimpleFilter': '#984ea3', 'Schumann14': '#ffff33', 'WSE2': '#f781bf', 'WSE1': '#999999'}
    
    
    #confusion_color_d = {'FN':'#c700fe', 'FP':'red', 'TP':'#00fe19', 'TN':'white'}
    
    rowLabels_d = {'WSE1':'Hydrodyn. (s1)', 'WSE2':'Hydrodyn. (s2)'}
    
    def __init__(self, 
                 run_name = None,
                 **kwargs):
 
        if run_name is None:
            run_name = 'post_v1'
        super().__init__(run_name=run_name, **kwargs)
    
    

    
       
class Plot_samples_wrkr(PostBase):
    
    def collect_samples_data(self, run_lib, 
                             sample_dx_fp=None,
                             **kwargs):
        """collect the filepaths from the run_lib
        
        Parameters
        ----------
        sample_dx_fp: str
            optional filepath to pickel of all simulations (so you dont have to read gpd each time)
        
        """
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('collect_samples_data',ext='.pkl', **kwargs)
        
        fp_lib={k:dict() for k in run_lib.keys()}
        metric_lib = {k:dict() for k in run_lib.keys()}
        log.info(f'on {len(run_lib)}')
        
 
        #=======================================================================
        # pull for each
        #=======================================================================
        for k0, d0 in run_lib.items(): #simulation name
            #print(dstr(d0))
            for k1, d1 in d0.items(): #cat0
                for k2, d2 in d1.items():
                    if k1=='vali':
                        if k2=='samp':
                            for k3, v3 in d2.items():
                                if k3=='samples_fp':
                                    fp_lib[k0][k3]=v3
 
                                else:
                                    metric_lib[k0][k3]=v3
                                    
        #=======================================================================
        # load frames
        #=======================================================================
        if sample_dx_fp is None:
            d=dict()
            true_serj = None
            for k0, d0 in fp_lib.items():
                for k1, fp in d0.items():
                    df = gpd.read_file(fp).drop('geometry', axis=1)
                    
                    true_ser = df['true']
                    d[k0]=df['pred']
                    
                    if not true_serj is None:
                        assert (true_ser==true_serj).all(), f'trues dont match {k0}{k1}'
                        
                    true_serj=true_ser
                    
            d['vali'] = true_ser
            dx = pd.concat(d, axis=1)
            
            #write compiled
            dx.to_pickle(ofp)
            log.info(f'wrote {str(dx.shape)} samples to \n    {ofp}')
            
        else:
            dx = pd.read_pickle(sample_dx_fp) 
                
        #=======================================================================
        # wrap
        #=======================================================================
        #log.info('got fp_lib:\n%s\n\nmetric_lib:\n%s'%(dstr(fp_lib), dstr(metric_lib)))
        log.info(f'finished on {str(dx.shape)}')
 
        return dx, metric_lib
                   
    def plot_samples_mat(self, 
                         df_raw, metric_lib,
                         figsize=None,
                         color_d=None,
                         col_keys = ['raw_hist', 'diff_hist', 'corr_scatter'],
                   add_subfigLabel=True,
                      transparent=True,
                      output_format=None,
                        **kwargs):
        """matrix plot comparing methods for downscaling: sampled values
        
        rows: 
            vali, methods
        columns:
            depth histogram, difference histogram, correlation plot
            
        same as Figure 5 on RICorDE paper"""
        
        #=======================================================================
        # defautls
        #=======================================================================
        if output_format is None: output_format=self.output_format
        
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('pSamplesMapt', ext='.'+output_format, **kwargs)
        
        log.info(f'on {df_raw.columns}')
        font_size=matplotlib.rcParams['font.size']
        
        if color_d is None: color_d = self.sim_color_d.copy()
        
        #=======================================================================
        # data prep
        #=======================================================================
        #drop any where the truth is zero
        bx = df_raw['vali']!=0
        log.info(f'dropped {bx.sum()}/{len(bx)} samples where vali=0')
        df = df_raw[bx]
 
        #=======================================================================
        # setup figure
        #=======================================================================
        row_keys = ['vali', 's14', 'cgs' ]   # list(df.columns)
        
        
        fig, ax_d = self.get_matrix_fig(row_keys, col_keys, logger=log,
                                        set_ax_title=False, figsize=figsize,
                                        constrained_layout=True,
                                        sharex='col',
                                        sharey='col',
                                        add_subfigLabel=add_subfigLabel,
                                        )
 
        #=======================================================================
        # plot loop---------
        #=======================================================================
 
        for rowk, d0 in ax_d.items():
            for colk, ax in d0.items():
                log.info(f'plotting {rowk} x {colk}')
                
                c = color_d[rowk]
                #txt_d = {rowk:''}
                txt_d=dict()
                
                hist_kwargs = dict(color=c, bins=30)
 
                #===============================================================
                # data prep
                #===============================================================
                #drop any that are zero
                bx = df[rowk]!=0
                assert bx.any()
                ser = df.loc[bx, rowk]
                
                #corresponding true values
                true_ser = df.loc[bx, 'vali']
                #===============================================================
                # raw histograms
                #===============================================================
                if colk == 'raw_hist':
 
                    n, bins, patches = ax.hist(ser, **hist_kwargs)
                    
                    stats_d = {k:getattr(ser, k)() for k in ['min', 'max', 'mean', 'count']}
                    txt_d.update(stats_d)
                    txt_d['bins'] = len(bins)
 
                elif rowk == 'vali':
                    hide_text(ax)
                    txt_d=None
                #===============================================================
                # difference histograms
                #===============================================================
                elif colk == 'diff_hist':
                    si = ser - true_ser
                    n, bins, patches = ax.hist(si, **hist_kwargs)
                    
                    #error calcs
                    #stats_d = {k:getattr(si, k)() for k in ['min', 'max', 'mean', 'count']}
                    with ErrorCalcs(pred_ser=ser, true_ser=true_ser, logger=log) as wrkr:
                        stats_d = wrkr.get_all(dkeys_l=['RMSE', 'bias', 'meanError'])
                    txt_d.update(stats_d)
                    #txt_d['bins'] = len(bins)
                    
                #===============================================================
                # scatter
                #===============================================================
                elif colk == 'corr_scatter':
                    xar, yar = true_ser.values, ser.values
                    xmin, xmax = xar.min(), xar.max()
                    
                    # scatters
                    ax.plot(xar, yar, color=c, linestyle='none', marker='.',
                            markersize=2, alpha=0.8)
                    
                    # 1:1
                    ax.plot([xmin, xmax], [xmin, xmax],
                            color='black', linewidth=1.0)
                    
                    # correlation
                    f, fit_d = self.scipy_lineregres(df.loc[bx, [rowk, 'vali']], xcoln='vali', ycoln=rowk)
                    
                    xar = np.linspace(xmin, xmax, num=10)
                    ax.plot(xar, f(xar), color='red', linewidth=1.0)
 
                    # post format
                    ax.grid()
                    
                    txt_d.update(fit_d)
                    
                #===============================================================
                # text
                #===============================================================
                if not txt_d is None:
                    ax.text(0.9, 0.1, get_dict_str(txt_d),
                                    transform=ax.transAxes, va='bottom', ha='right',
                                     fontsize=font_size, color='black',
                                     bbox=dict(boxstyle="round,pad=0.3", fc="white", lw=0.0, alpha=0.5),
                                     )
                 
        #=======================================================================
        # post format--------
        #=======================================================================
 
        for rowk, d0 in ax_d.items():
            for colk, ax in d0.items():
 
                        
                
                #first col
                if colk==col_keys[0]:
                    ax.set_ylabel('count')
                    
                    
                #last col
                if colk==col_keys[-1]:
                    if not rowk=='vali':
                        ax.set_ylabel('pred. depth (m)')
                        
 
                    
                    ax.text(1.1, 0.5, nicknames_d2[rowk],
                                    transform=ax.transAxes, va='center', ha='center',
                                     fontsize=font_size+2, color='black',rotation=-90,
                                     #bbox=dict(boxstyle="round,pad=0.3", fc="white", lw=0.0, alpha=0.5),
                                     )
                    
                #first row
                if rowk==row_keys[0]:
                    pass
                    
                    #===========================================================
                    # ax.set_title({
                    #     'raw_hist':'depths',
                    #     'diff_hist':'differences',
                    #     'corr_scatter':'correlation'
                    #     }[colk])
                    # 
                    if not colk=='raw_hist':
                        ax.axis('off')
                    
                #last row
                if rowk==row_keys[-1]:
                    ax.set_xlabel({
                        'raw_hist':'depths (m)',
                        'diff_hist':'pred. - true (m)',
                        'corr_scatter':'true depths (m)'
                        }[colk])
 
                    
        #=======================================================================
        # wrap
        #=======================================================================
        log.info('finished')
        return self.output_fig(fig, ofp=ofp, logger=log, dpi=600, transparent=transparent)
    
    def scipy_lineregres(self,
               df_raw,
 
                       xcoln='area',
                       ycoln='gvimp_gf',
 
 
 
               ):
        
        #=======================================================================
        # defaults
        #=======================================================================
 
        log = self.logger.getChild('scipy_lineregres')
        
        #=======================================================================
        # setup data
        #=======================================================================
        df1 = df_raw.loc[:, [xcoln, ycoln]].dropna(how='any', axis=0)
        xtrain, ytrain=df1[xcoln].values, df1[ycoln].values
        #=======================================================================
        # scipy linregress--------
        #=======================================================================
        lm = scipy.stats.linregress(xtrain, ytrain)
        
        predict = lambda x:np.array([lm.slope*xi + lm.intercept for xi in x])
        
        
        return predict, {'rvalue':lm.rvalue, 'slope':lm.slope, 'intercept':lm.intercept}
    
 
class PostSession(Plot_inun_peformance, Plot_samples_wrkr, Plot_hyd_HWMS,
                   ValidateSession):
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
 