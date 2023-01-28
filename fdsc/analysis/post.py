'''
Created on Jan. 9, 2023

@author: cefect

data analysis on multiple downscale results
'''


import logging, os, copy, datetime, pickle, pprint
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

from hp.plot import Plotr, get_dict_str, hide_text
from hp.pd import view
from hp.rio import (    
    get_ds_attr,get_meta
    )
from hp.err_calc import get_confusion_cat


 
from fdsc.analysis.valid import ValidateSession
from fdsc.base import nicknames_d

nicknames_d['validation']='vali'
nicknames_d2 = {v:k for k,v in nicknames_d.items()}

def dstr(d):
    return pprint.pformat(d, width=30, indent=0.3, compact=True, sort_dicts =False)

class Plot_rlays_wrkr(object):
    gdf_d=dict() #container for preloading geopandas
    
    def collect_rlay_fps(self, run_lib, **kwargs):
        """collect the filepaths from the run_lib"""
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('collect_rlay_fps', **kwargs)
        
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
                            dep2=d2
                        elif k2 in ['dem1']:
                            dem_fp = d2                    
                        
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
                        
                        #get the sample points layer
                        elif k2=='samp':
                            for k3, v3 in d2.items():
                                if k3=='samples_fp':
                                    fp_lib[k0][k3]=v3
                            
                    else:
                        pass
 
        #=======================================================================
        # get validation
        #=======================================================================
        fp_lib = {**{'vali':{'dep1':dep1V, 'dem1':dem_fp, 'dep2':dep2}}, **fp_lib} #order matters
 
 
        log.info('got fp_lib:\n%s\n\nmetric_lib:\n%s'%(dstr(fp_lib), dstr(metric_lib)))
        
        self.fp_lib=fp_lib
        return fp_lib, metric_lib
        
    



    def plot_rlay_mat(self,
                      fp_lib, metric_lib=None, 
                      figsize=(9,9),
 
            **kwargs):
        """matrix plot comparing methods for downscaling: rasters
        
        TODO: add point samples (color coded by confusion)
        
        rows: cols
            valid: 
            methods
        columns
            depthRaster r2, depthRaster r1, confusionRaster
        """
        #=======================================================================
        # defaults
        #=======================================================================
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('rlayMat',ext='.png', **kwargs)
        
        log.info(f'on {list(fp_lib.keys())}')
        
        cc_d = self.confusion_codes.copy()
        
        #spatial meta from dem for working with points
        self.rmeta_d = get_meta(fp_lib['vali']['dem1']) 
 
        #=======================================================================
        # setup figure
        #=======================================================================
        row_keys = list(fp_lib.keys())
        col_keys = ['c1', 'c2', 'c3']
        
        #grid_lib={k:dict() for k in row_keys}
        grid_lib=dict()
        
        for rowk in row_keys:
            if rowk=='vali':
                col_d = {'c1':None, 'c2':'dep1', 'c3':'dep2'}
            else:
                col_d = {'c1':'pie', 'c2':'dep1', 'c3':'confuGrid_fp'}
 
                
            grid_lib[rowk] = col_d             
                
        log.info('on %s'%dstr(grid_lib))
 
        fig, ax_d = self.get_matrix_fig(row_keys, col_keys, logger=log, 
                                        set_ax_title=False, figsize=figsize,
                                        constrained_layout=True,
                                        add_subfigLabel=True,
                                        )
        
 
        
        #=======================================================================
        # colormap
        #=======================================================================
        confusion_color_d = {
            'FN':'#c700fe', 'FP':'#ff5101', 'TP':'#00fe19', 'TN':'white'
            }
        
        #get rastetr val to color conversion for confusion grid
        cval_d = {v:confusion_color_d[k] for k,v in cc_d.items()}        
        cval_d = {k:cval_d[k] for k in sorted(cval_d)} #sort it
 
        #build a custom color map        
        confuGrid_cmap = matplotlib.colors.ListedColormap(cval_d.values())        
        confuGrid_norm = matplotlib.colors.BoundaryNorm(
                                            np.array([0]+list(cval_d.keys()))+1, #bounds tt capture the data 
                                              ncolors=len(cval_d),
                                              #cmap.N, 
                                              extend='neither',
                                              clip=True,
                                              )
        
        #=======================================================================
        # plot loop------
        #=======================================================================
        axImg_d = dict() #container for objects for colorbar
        #dep1_yet=False
        for rowk, d0 in ax_d.items():
            for colk, ax in d0.items():                
                gridk = grid_lib[rowk][colk]
                aname = nicknames_d2[rowk]                
                log.debug(f'plot loop for {rowk}.{colk}.{gridk} ({aname})')
                
 
                #===============================================================
                # blank
                #===============================================================
                if gridk is None:
                    ax.set_axis_off()
                    hide_text(ax)
 
                    continue
                #===============================================================
                # raster plot
                #===============================================================
                elif not gridk=='pie':
                    assert gridk in fp_lib[rowk], f'missing data file for {rowk}.{colk}.{gridk} ({aname})'
                    fp = fp_lib[rowk][gridk]
                    
                    log.info(f'plotting {rowk} x {colk} ({gridk}): {os.path.basename(fp)}')
                    with rio.open(fp, mode='r') as ds:
                        ar_raw = ds.read(1, window=None, masked=True)
                        
                        #===========================================================
                        # #apply masks
                        #===========================================================
                        if 'dep' in gridk:
                            ar = np.where(ar_raw==0, np.nan, ar_raw)
                        elif 'confuGrid' in gridk:
                            #mask out true negatives
                            ar = np.where(ar_raw==cc_d['TN'], np.nan, ar_raw)
                        elif 'dem1' ==gridk:
                            ar = np.where(ar_raw<130, ar_raw, np.nan)
                            print(ar_raw.max())
                        else:
                            raise KeyError(gridk)
                            
                        #===========================================================
                        # #get styles by key
                        #===========================================================
                        if 'confuGrid_fp' ==gridk:
                            cmap=confuGrid_cmap
                            norm=confuGrid_norm
                        elif gridk=='dem1':
                            cmap='plasma'
                            norm=None
                        elif 'dep' in gridk:
                            cmap='viridis'
                            norm=None
                        else:
                            raise KeyError(gridk)
                         
                        #===========================================================
                        # #raster plot
                        #===========================================================
                        _ = show(ar, 
                                      transform=ds.transform, 
                                      ax=ax,contour=False, cmap=cmap, interpolation='nearest', 
                                      norm=norm)
                        
                        #ax_img=ax.imshow(ar, cmap=cmap, interpolation='nearest', norm=norm, aspect='equal')
                        
                        #=======================================================
                        # asset sample plot
                        #=======================================================
                        if colk =='c2' and 'samples_fp' in fp_lib[rowk]:                           
 
                            #load
                            gdf = self._load_gdf(rowk)
                            
                            #drop Trues 
                            gdf1 = gdf.loc[~gdf['confusion'].isin(['TN', 'TP']), :]
                            
                            #map colors                            
                            gdf1['conf_color'] = gdf1['confusion'].replace(cc_d)                            
                            
                            #plot
                            _= gdf1.plot(column='conf_color', ax=ax, cmap=confuGrid_cmap, norm=confuGrid_norm,
                                     markersize=.2, marker='.', #alpha=0.8,
                                     )
                        
 
                        #===========================================================
                        # post format
                        #===========================================================
                        #hide labels
                        ax.get_xaxis().set_ticks([])
                        ax.get_yaxis().set_ticks([])
                        
                        #add text
                        if gridk=='confuGrid_fp' and isinstance(metric_lib, dict):
                            md = {k:v for k,v in metric_lib[rowk].items() if not k in cc_d.keys()}
                            #md = {**{rowk:''}, **md} 
                            ax.text(0.98, 0.05, get_dict_str(md), transform=ax.transAxes, 
                                    va='bottom', ha='right', fontsize=6, color='black',
                                    bbox=dict(boxstyle="round,pad=0.3", fc="white", lw=0.0,alpha=0.5 ),
                                    )
                            
                        #colorbar
                        if not gridk in axImg_d:
                            axImg_d[gridk]=[obj for obj in ax.get_children() if isinstance(obj, AxesImage)][0]
                
                #===============================================================
                # pie plot         
                #===============================================================
                elif gridk=='pie':
                    gdf = self._load_gdf(rowk)
                    
                    total_ser = gdf['confusion'].value_counts() #.rename(nicknames_d2[rowk])
                    colors_l = [confusion_color_d[k] for k in total_ser.index]
 
                    ax.pie(total_ser.values, colors=colors_l, autopct='%1.1f%%', 
                           shadow=True, labels=total_ser.index.values)
                        
 
        #=======================================================================
        # colorbar-------
        #=======================================================================
        for rowk, d0 in ax_d.items():
            for colk, ax in d0.items():
                #only last row 
                if not rowk==row_keys[-1]:
                    continue
                               
                gridk = grid_lib[rowk][colk]
                
                if gridk is None:
                    gridk='dem1'
                
                            
                if 'dep' in gridk:
                    spacing='proportional'
                    label='WSH (m)'
                    fmt = matplotlib.ticker.FuncFormatter(lambda x, p:'%.1f' % x)
                    location='bottom'
                    #cax = cax_top
                elif 'confuGrid' in gridk:
 
                    #spacing='proportional'
                    spacing='uniform'
                    label='Confusion'
                    fmt=None
                    #fmt = matplotlib.ticker.FuncFormatter(lambda x, p:cc_di[x])
                    #cax=cax_bot
                    location='bottom'
                    
                elif 'dem1'==gridk:
                    spacing='proportional'
                    label='DEM (masl)'
                    fmt = matplotlib.ticker.FuncFormatter(lambda x, p:'%.0f' % x)
                    location='bottom'
                    
                elif gridk=='pie':
                    continue
                    
                else:
                    raise KeyError(gridk)
                
                cbar = fig.colorbar(axImg_d[gridk],
                                #cax=cax, 
 
                                orientation='horizontal',
                         ax=ax, location=location, # steal space from here
                         extend='both', #pointed ends
                         format=fmt, label=label,spacing=spacing,
                         shrink=0.8,
                         )
                
                #relabel
                if 'confuGrid' in gridk:
                    #help(cbar)
                    #print(cbar.get_ticks()) #[101, 102, 111, 112]
                    #print(cc_d)
                    cbar.set_ticks([(101-1)/2+1, 101.5, (111-102)/2+102, 111.5], 
                                   labels = [{v:k for k,v in cc_d.items()}[k0] for k0 in cval_d.keys()]
                                   #labels=list(cval_d.keys())
                                   )
                    #cbar.set_ticklabels(list(cc_d.keys()))
                     
                
 
            
        #=======================================================================
        # post format-------
        #=======================================================================
 
        for rowk, d0 in ax_d.items():
            for colk, ax in d0.items():
                #turn off useless axis
                
                #first col
                if colk==col_keys[0]:
                    ax.set_ylabel(nicknames_d2[rowk], 
                                  #fontsize=6,
                                  )
                #second col
                if colk==col_keys[1]:
                    pass
                    #ax.set_ylabel(nicknames_d2[rowk], fontsize=6)
                    
                #first row
                if rowk==row_keys[0]:
                    pass
                    #===========================================================
                    # ax.set_title({
                    #     'dep1':'WSH (r02)',
                    #     'dep2':'WSH (r32)',
                    #     'confuGrid_fp':'confusion'
                    #     }[colk])
                    #===========================================================
 
                    
        #=======================================================================
        # wrap
        #=======================================================================
        log.info('finished')
        return self.output_fig(fig, ofp=ofp, logger=log)
    
    def _load_gdf(self, dkey, samples_fp=None, rmeta_d=None):
        """convenienve to retrieve pre-loaded or load points"""
        #=======================================================================
        # defaults
        #=======================================================================
        if rmeta_d is None: rmeta_d=self.rmeta_d.copy()
        
        if samples_fp is None:
            samples_fp = self.fp_lib[dkey]['samples_fp']
        
        #=======================================================================
        # preloaded
        #=======================================================================
        if dkey in self.gdf_d:
            gdf = self.gdf_d[dkey].copy()
            
        #=======================================================================
        # load
        #=======================================================================
        else:        
            gdf = gpd.read_file(samples_fp, bbox=rmeta_d['bounds'])
            
            #compute wet-dry confusion
            """TODO: move this to valid.py"""
            
            gdf['confusion'] = get_confusion_cat(gdf['true']>0.0, gdf['pred']>0.0)
            


        #=======================================================================
        # check
        #=======================================================================
        assert gdf.crs == rmeta_d['crs']
        
        return gdf 
 
        
class Plot_samples_wrkr(object):
    
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
                    
            d['true'] = true_ser
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
                         df, metric_lib,
                         figsize=None,
                         color_d=None,
                        **kwargs):
        """matrix plot comparing methods for downscaling: sampled values
        
        rows: 
            vali, methods
        columns:
            depth histogram, difference histogram, correlation plot
            
        same as Figure 5 on RICorDE paper"""
        
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('pSamplesMapt',ext='.svg', **kwargs)
        
        log.info(f'on {df.columns}')
        
        if color_d is None: color_d = self.sim_color_d.copy()
        
 
        #=======================================================================
        # setup figure
        #=======================================================================
        row_keys = ['true', 'nodp', 'cgs', 'bgl']#list(df.columns)
        col_keys = ['raw_hist', 'diff_hist', 'corr_scatter']
        
        
        fig, ax_d = self.get_matrix_fig(row_keys, col_keys, logger=log, 
                                        set_ax_title=False, figsize=figsize,
                                        constrained_layout=True,
                                        sharex='col', 
                                        sharey='col',
                                        add_subfigLabel=True,
                                        )
        
 
        #=======================================================================
        # plot loop
        #=======================================================================
 
        for rowk, d0 in ax_d.items():
            for colk, ax in d0.items():
                log.info(f'plotting {rowk} x {colk}')
                ser = df[rowk]
                c=color_d[rowk]
                txt_d={rowk:''}
                
                
                hist_kwargs = dict(color=c, bins=30)
                """:
                plt.show()
                """
                #===============================================================
                # raw histograms
                #===============================================================
                if colk=='raw_hist':
 
                    n, bins, patches = ax.hist(ser, **hist_kwargs)
                    
                    stats_d = {k:getattr(ser,k)() for k in ['min', 'max', 'mean', 'count']}
                    txt_d.update(stats_d)
                    txt_d['bins']=len(bins)
 
                elif rowk=='true':
                    continue
                    
                #===============================================================
                # difference histograms
                #===============================================================
                elif colk=='diff_hist':
                    si = ser-df['true']
                    n, bins, patches = ax.hist(si, **hist_kwargs)
                    
                    stats_d = {k:getattr(si,k)() for k in ['min', 'max', 'mean', 'count']}
                    txt_d.update(stats_d)
                    txt_d['bins']=len(bins)
                    
                #===============================================================
                # scatter
                #===============================================================
                elif colk=='corr_scatter':
                    xar, yar = df['true'].values, ser.values
                    xmin, xmax = xar.min(), xar.max()
                    
                    #scatters
                    ax.plot(xar, yar, color=c, linestyle='none', marker='.',
                            markersize=2, alpha=0.8)
                    
                    #1:1
                    ax.plot([xmin, xmax], [xmin, xmax],
                            color='black', linewidth=1.0)
                    
                    #correlation
                    f, fit_d = self.scipy_lineregres(df.loc[:, [rowk, 'true']],xcoln='true', ycoln=rowk)
                    
                    xar =  np.linspace(xmin, xmax, num=10)
                    ax.plot(xar, f(xar), color='red', linewidth=1.0)
 
                    #post format
                    ax.grid()
                    
                    txt_d.update(fit_d)
                    
                #===============================================================
                # text
                #===============================================================
                ax.text(0.9, 0.1, get_dict_str(txt_d), 
                                transform=ax.transAxes, va='bottom',ha='right',
                                 fontsize=8, color='black',
                                 bbox=dict(boxstyle="round,pad=0.3", fc="white", lw=0.0,alpha=0.5 ),
                                 )
                 
        #=======================================================================
        # post format
        #=======================================================================
 
        for rowk, d0 in ax_d.items():
            for colk, ax in d0.items():
 
                        
                
                #first col
                if colk==col_keys[0]:
                    ax.set_ylabel('count')
                    
                    
                #last col
                if colk==col_keys[-1]:
                    ax.set_ylabel('pred. depth (m)')
                    
                #first row
                if rowk==row_keys[0]:
                    
                    ax.set_title({
                        'raw_hist':'depths',
                        'diff_hist':'differences',
                        'corr_scatter':'correlation'
                        }[colk])
                    
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
        return self.output_fig(fig, ofp=ofp, logger=log)
    
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
        

class PostSession(Plot_rlays_wrkr, Plot_samples_wrkr, 
                  Plotr, ValidateSession):
    sim_color_d = {'true':'black', 'nodp':'orange', 'cgs':'teal', 'bgl':'violet'}
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
            assert os.path.exists(fp), f'bad filepath for \'{k}\':\n    {fp}'
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
        
    

            
        
        
        
        
 
