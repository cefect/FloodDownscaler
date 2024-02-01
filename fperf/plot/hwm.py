'''
Created on Mar. 20, 2023

@author: cefect
'''

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
import earthpy as et
import earthpy.spatial as es
import earthpy.plot as ep

from hp.plot import Plotr, get_dict_str, hide_text, cm
from hp.pd import view
from hp.rio import (    
    get_ds_attr,get_meta, SpatialBBOXWrkr
    )
from hp.err_calc import get_confusion_cat, ErrorCalcs
from hp.gpd import get_samples
from hp.fiona import get_bbox_and_crs

from fperf.plot.base import PostBase



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
    
class Plot_HWMS(PostBase):
    """plotter for HWM analysis"""
    
    def collect_HWM_data(self, serx, pick_fp=None,
                         write=False,
                         **kwargs):
        """load HWM gdfs from passed serx
        
        Pars
        --------
        pick_fp: str, None
            optional filepath to load teh compiled data from
            
        """
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('collect_HWM', **kwargs)
        
        #=======================================================================
        # build--------
        #=======================================================================
        if pick_fp is None:
            #=======================================================================
            # loop and load each prediction
            #=======================================================================
            df_d = dict()
            for simName, fp in serx.loc[idx[:, 'pred_hwm_fp']].items():
                gdf = gpd.read_file(fp)
                df_d[simName] = gdf['pred']
                
            #=======================================================================
            # load observed
            #=======================================================================
            df_d['true'] = gdf['true'] #use last
            all_gdf = pd.concat(df_d, axis=1)
            
            #add geo
            all_gdf = all_gdf.set_geometry(gdf.geometry)
            
            #===================================================================
            # write it
            #===================================================================
            if write:
                self._write_pick(all_gdf, resname='hwm_gdf', out_dir=tmp_dir)
 
            
        #=======================================================================
        # compiled-------
        #=======================================================================
        else:
            assert not write
            log.warning(f'loading from \n    {pick_fp}')
            with open(pick_fp, 'rb') as f:
                all_gdf = pickle.load(f)
            
        
        
        #=======================================================================
        # wrap
        #=======================================================================

                
        log.info(f'loaded {str(all_gdf.shape)}')
        return all_gdf
 
        
    

    def _ax_hwm_scatter(self,
                        ax, xar, yar,                         
                        
                        style_d=dict(),
                         xlim=None,  
                         max_val=None,
                         metaLabel=None,
                         metaKeys_l = None,
                         ):
        """add a scatter for HWMs
        
        Pars
        -----------
        metaLabel: str, default none
            label to add at head of meta text
            
        metaKeys_l: list
            list of metrics to ionclude in meta text
        """
            
        #=======================================================================
        # defaults
        #=======================================================================
        if xlim is None: xlim=min(xar)
        if max_val is None: max_val=max(xar)
        if metaKeys_l is None: metaKeys_l = [
            #'pearson', #same as rvalue
            'rvalue', 'stderr', 'rmse']
        #=======================================================================
        # ploat
        #=======================================================================
        ax.plot(xar, yar, linestyle='none', **style_d)
        #=======================================================================
        # compute error
        #=======================================================================
        slope, intercept, rvalue, pvalue, stderr = scipy.stats.linregress(xar, yar)
        pearson, pval = scipy.stats.pearsonr(xar, yar)
        rmse = math.sqrt(np.square(xar - yar).mean())
        x_vals = np.array(xlim)
        y_vals = intercept + slope * x_vals
        
        #collect meta metrics
        meta_all_d = dict(
                            pearson=pearson, #same as r value
                            pval=pval,
                          rvalue=rvalue,
                          pvalue=pvalue,stderr=stderr, rmse=rmse)
        assert set(metaKeys_l).difference(meta_all_d.keys())==set(), 'requested some bad keys'
        meta_d = {k:v for k,v in meta_all_d.items() if k in metaKeys_l} #filter to request
 
        #===================================================================
        # plot correlation
        #===================================================================
        ax.plot(x_vals, y_vals, color=style_d['color'], linewidth=0.75, label='regression')
        #1:1 line
        ax.plot([0.0, max_val * 1.1], [0.0, max_val * 1.1], color='black', label='1:1', linewidth=0.5, linestyle='dashed')
        #===========================================================
        # text
        #===========================================================
        if not metaLabel is None:
            md = {**{metaLabel:''}, **meta_d}
        else:
            md = meta_d
            
            
        ax.text(0.98, 0.05, get_dict_str(md, num_format='{:.3f}'), 
            transform=ax.transAxes, 
            va='bottom', ha='right', 
            #fontsize=font_size,
            color='black', 
            bbox=dict(boxstyle="round,pad=0.3", fc="white", lw=0.0, alpha=0.5))
        #=======================================================================
        # post
        #=======================================================================
        #make square
        ax.set_ylim(0, max_val)
        ax.set_xlim(0, max_val)
        ax.set_aspect('equal', adjustable='box')
        
        return meta_d

    def plot_HWM(self, gdf, metric_lib=None,
                 output_format=None,
                 figsize=None,
                 transparent=False,
                 style_d = {
                          'WSE1':dict(color='#996633', label='fine (s1)', marker='x'),
                          'WSE2':dict(color='#9900cc', label='coarse (s2)', marker='o', fillstyle='none')
                                 },
                 
                 xlim=None,
                 **kwargs):
        
        """matrix of scatter plots for performance against HWMs"""
        
        #=======================================================================
        # defaults
        #=======================================================================
        if output_format is None: output_format=self.output_format
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('pHWM', ext='.'+output_format, **kwargs)
        
        
        #=======================================================================
        # setup figure
        #=======================================================================
        row_keys = gdf.drop(['true','geometry'], axis=1).columns.tolist() #['CostGrow', 'Basic', 'SimpleFilter', 'Schumann14', 'WSE2', 'WSE1']
        
        col_keys = ['scatter']
        
        fig, ax_d = self.get_matrix_fig(row_keys, col_keys, logger=log,
                                set_ax_title=False, figsize=figsize,
                                constrained_layout=True,
                                sharex='col',
                                sharey='col',
                                add_subfigLabel=True,
                                figsize_scaler=3,
                                )
        
        #add any missing to the style d
        for k in row_keys:
            if not k in style_d:
                style_d[k] = dict()
                
        #=======================================================================
        # data prep
        #=======================================================================
        true_ser = gdf['true']
        
        max_val = math.ceil(gdf.max().max())
        if xlim is None:
            xlim = (0, max_val)
        #=======================================================================
        # plot loop------
        #=======================================================================
        
        meta_lib=dict()
        for rowk, d0 in ax_d.items():
            for colk, ax in d0.items():
                log.info(f'plotting {rowk}x{colk}')
                
                if colk=='scatter': 
 
                    #scatter
                    xar, yar = true_ser.values, gdf[rowk].values
                    self._ax_hwm_scatter(ax, xar, yar, style_d=style_d[rowk], xlim=xlim, max_val=max_val, rowk=rowk)
                    
                    
                    
                    
        #=======================================================================
        # post
        #=======================================================================
        
        for rowk, d0 in ax_d.items():
            for colk, ax in d0.items():
                
                #first row
                if rowk==row_keys[0]:
                    #last columns
                    if colk==col_keys[-1]:
                        pass
                        #ax.legend()
                        
                #first col
                if colk==col_keys[0]:
 
                    ax.set_ylabel('simulated max depth (m)')
                    
                    #last row
                    if rowk==row_keys[-1]:
                        ax.set_xlabel('observed HWM (m)')
                    
        
        #=======================================================================
        # wrap
        #=======================================================================
        return self.output_fig(fig, logger=log, ofp=ofp, transparent=transparent)
                
                   




    def _get_fig_mat_models(self, mod_keys, 
                            ncols=1,  
                            total_fig_width=None, 
                            figsize=None, 
                            constrained_layout=True, 
                            **kwargs):
        #=======================================================================
        # defaults
        #=======================================================================
 
        if total_fig_width is  None: total_fig_width = matplotlib.rcParams['figure.figsize'][0]
        #=======================================================================
        # #reshape into a frame
        #=======================================================================
        #bad division
        mod_keys2 = copy.copy(mod_keys)
 
        if len(mod_keys) % ncols != 0:
            for i in range(ncols - len(mod_keys) % ncols):
                mod_keys2.append(np.nan)
            
 
        
        mat_df = pd.DataFrame(np.array(mod_keys2).reshape(-1, ncols))
        mat_df.columns = [f'c{e}' for e in mat_df.columns]
        mat_df.index = [f'r{e}' for e in mat_df.index]
        row_keys = mat_df.index.tolist()
        
        col_keys = mat_df.columns.tolist()
        
        #figure size
        if figsize is None:
            figsize_scaler = (total_fig_width / ncols)
        else:
            #assert ncols is None
            assert total_fig_width is None
            figsize_scaler = None
            
        fig, ax_d = self.get_matrix_fig(row_keys, col_keys, 
            set_ax_title=False, 
            figsize=figsize, 
            constrained_layout=constrained_layout, 
            sharex='all', 
            sharey='all', 
            #add_subfigLabel=True, 
            figsize_scaler=figsize_scaler, 
            **kwargs)
        
        return ax_d, mat_df, row_keys, col_keys, fig

    def plot_HWM_scatter(self, 
                         gdf, 
 
                     
                 output_format=None,
                  
                 transparent=False,
                 style_d = {},
                 style_default_d=dict(marker='x', fillstyle='none', alpha=0.8),
                 color_d=None,
                 mod_keys=None,
                 rowLabels_d=None,metaKeys_l=None,
                 xlim=None,
                 fig_mat_kwargs=dict(figsize=None,ncols=1,total_fig_width=14),
                 **kwargs):
        
        """matrix of scatter plots for performance against HWMs...by column count
        
        Pars
        ----------
        gdf: dataFrame
            cols: simNames. vals: HWM samples
        """
        
        #=======================================================================
        # defaults
        #=======================================================================
        if output_format is None: output_format=self.output_format
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('hwmScatter', ext='.'+output_format, **kwargs)
        
        #list of model values
        if mod_keys is None:
            mod_keys = gdf.drop(['true','geometry'], axis=1).columns.tolist()
        
        assert set(mod_keys).difference(gdf.columns)==set()
        
        if rowLabels_d is None:
            rowLabels_d=self.rowLabels_d
        
            
        if color_d is  None: 
            #color_d = self.sim_color_d.copy()
            color_d = self._build_color_d(mod_keys, cmap = plt.cm.get_cmap(name='Dark2'))
 
        #=======================================================================
        # setup figure
        #=======================================================================        
        ax_d, mat_df, row_keys, col_keys, fig = self._get_fig_mat_models(
                                            mod_keys,logger=log, **fig_mat_kwargs)
        
        #=======================================================================
        # setup style
        #======================================================================= 

            
        #add any missing to the style d
        for k in mod_keys:
            if not k in style_d:
                style_d[k] = style_default_d.copy()
                
            if not 'color' in style_d[k]:
                style_d[k]['color'] = color_d[k]
                
        #=======================================================================
        # data prep
        #=======================================================================
        true_ser = gdf['true']
        
        max_val = math.ceil(gdf.max().max())
        if xlim is None:
            xlim = (0, max_val)
        #=======================================================================
        # plot loop------
        #=======================================================================
        
        meta_lib=dict()
        for rowk, d0 in ax_d.items():
            for colk, ax in d0.items():                
                #===============================================================
                # setup
                #===============================================================
                modk = mat_df.loc[rowk, colk]
                if modk=='nan': 
                    ax.axis('off')
                    continue
                
                log.info(f'plotting {rowk}x{colk} ({modk})')
                
                #get labels
                if modk in rowLabels_d:
                    metaLabel = rowLabels_d[modk]
                else:
                    metaLabel=modk
 
                #get data
                xar, yar = true_ser.values, gdf[modk].values
                #===============================================================
                # #scatter
                #===============================================================
                
                meta_lib[modk] = self._ax_hwm_scatter(ax, xar, yar, style_d=style_d[modk], 
                                                      xlim=xlim, max_val=max_val, metaLabel=metaLabel,
                                                      metaKeys_l=metaKeys_l)
 
        #=======================================================================
        # post
        #=======================================================================
        
        for rowk, d0 in ax_d.items():
            for colk, ax in d0.items():
                
                #first row
                if rowk==row_keys[0]:
                    #last columns
                    if colk==col_keys[-1]:                        
                        ax.legend(loc=9)
                        
                        """
                        plt.show()
                        """
                        
                #first col
                if colk==col_keys[0]:
 
                    ax.set_ylabel('simulated max depth (m)')
                    
                #last row
                if rowk==row_keys[-1]:
                    ax.set_xlabel('observed HWM (m)')
                    
        
        #=======================================================================
        # wrap
        #=======================================================================
        return self.output_fig(fig, logger=log, ofp=ofp, transparent=transparent)


