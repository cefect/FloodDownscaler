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

from hp.plot import Plotr, get_dict_str, hide_text
from hp.pd import view
from hp.rio import (    
    get_ds_attr,get_meta
    )
from hp.err_calc import get_confusion_cat, ErrorCalcs
from hp.gpd import get_samples


 
from fdsc.analysis.valid.v_ses import ValidateSession
from fdsc.base import nicknames_d

#nicknames_d['Hydrodynamic (s1)']='WSE1'
cm = 1/2.54
 
#nicknames_d2 = {v:k for k,v in nicknames_d.items()}


def dstr(d):
    return pprint.pformat(d, width=30, indent=0.3, compact=True, sort_dicts =False)

class Plot_rlays_wrkr(object):
    gdf_d=dict() #container for preloading geopandas
    
    def collect_rlay_fps(self, run_lib, **kwargs):
        """collect the filepaths from the run_lib"""
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('collect_rlay_fps', **kwargs)
        
        fp_lib={k:dict() for k in run_lib.keys()}
        metric_lib = fp_lib.copy()
        
 
        #=======================================================================
        # grid filepaths
        #=======================================================================
 
        for k0, d0 in run_lib.items(): #simulation name
 
            fp_d = d0['vali']['fps']            
            
            #clean
            fp_lib[k0] = {k.replace('_fp', ''):d for k,d in fp_d.items() if not d is None}
            
        log.info(f'collected filepaths on {len(fp_lib)} sims\n%s'%dstr(fp_lib))
        
        #=======================================================================
        # metrics
        #=======================================================================
        for k0, d0 in run_lib.items(): #simulation name
            with open(d0['smry']['valiMetrics_fp'], 'rb') as f:
                valiMetrics_d = pickle.load(f)
                assert isinstance(valiMetrics_d, dict)
                
            metric_lib[k0]=valiMetrics_d['inun']
            
        log.info(f'collected inundation metrics on {len(metric_lib)} sims')#\n%s'%dstr(metric_lib))
 
        self.fp_lib=fp_lib
        return fp_lib, metric_lib



    def plot_rlay_mat(self,
                      fp_lib, metric_lib=None,
                      figsize=None, #(12, 9),
                      row_keys=None,col_keys = None,
                      add_subfigLabel=True,
                      transparent=True,
                      font_size=None,
                      confusion_color_d=None,
                      output_format=None,
                      rowLabels_d = {'WSE1':'Hydrodyn. (s1)'},
 
            **kwargs):
        """matrix plot comparing methods for downscaling: rasters
 
        rows: cols
            valid: 
            methods
        columns
            depthRaster r2, depthRaster r1, confusionRaster
            
        Pars
        --------
        fp_lib: dict
            {row key i.e. method name: {gridName: filepath}}
        """
        #=======================================================================
        # defaults
        #=======================================================================
        if output_format is None: output_format=self.output_format
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('rlayRes', ext='.'+output_format, **kwargs)
        
        log.info(f'on {list(fp_lib.keys())}')
        
        if font_size is None:
            font_size=matplotlib.rcParams['font.size']
        if confusion_color_d is None:
            confusion_color_d=self.confusion_color_d.copy()
        
        cc_d = self.confusion_codes.copy()
        
        #spatial meta from dem for working with points
        self.rmeta_d = get_meta(fp_lib['WSE1']['dem']) 
 
        #=======================================================================
        # setup figure
        #=======================================================================
        if row_keys is None:
            row_keys = ['WSE1', 'Basic', 'SimpleFilter', 'CostGrow', 'Schumann14'] #list(fp_lib.keys())
            
        assert set(row_keys).difference(fp_lib.keys())==set()
            
        if col_keys is None:
            col_keys = ['c2', 'c3']
 
        grid_lib=dict()
        
        #specify the axis key for each row
        for rowk in row_keys:
            #===================================================================
            # if rowk=='vali':
            #     col_d = {'c1':None, 'c2':'dep1', 'c3':'dep2'}
            # else:
            #     col_d = {'c1':'pie', 'c2':'dep1', 'c3':'confuGrid_fp'}
            #===================================================================                
            grid_lib[rowk] = {'c2':'pred_wd', 'c3':'confuGrid'}             
                
        log.info('on %s'%dstr(grid_lib))
 
        fig, ax_d = self.get_matrix_fig(row_keys, col_keys, logger=log, 
                                        set_ax_title=False, figsize=figsize,
                                        constrained_layout=True,
                                        add_subfigLabel=add_subfigLabel,
                                        )
        
 
        
        #=======================================================================
        # colormap
        #=======================================================================        
        
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
                #aname = nicknames_d2[rowk]
                aname=rowk                
                log.debug(f'plot loop for {rowk}.{colk}.{gridk} ({aname})')
                
 
                #===============================================================
                # blank
                #===============================================================
                if gridk is None:
                    ax.set_axis_off()
                    hide_text(ax) 
                    continue
                #===============================================================
                # raster plot-----
                #===============================================================
                if gridk in ['pred_wd', 'confuGrid']:
                    assert rowk in fp_lib, rowk
                    assert gridk in fp_lib[rowk], f'missing data file for {rowk}.{colk}.{gridk} ({aname})'
                    fp = fp_lib[rowk][gridk]
                    
                    log.info(f'plotting {rowk} x {colk} ({gridk}): {os.path.basename(fp)}')
                    with rio.open(fp, mode='r') as ds:
                        ar_raw = ds.read(1, window=None, masked=True)
                        
                        #===========================================================
                        # #apply masks
                        #===========================================================
                        if ('dep' in gridk) or ('wd' in gridk):
                            assert np.any(ar_raw==0), 'depth grid has no zeros ' + rowk
                            ar = np.where(ar_raw==0, np.nan, ar_raw)
                        elif 'confuGrid' in gridk:
                            #mask out true negatives
                            ar = np.where(ar_raw==cc_d['TN'], np.nan, ar_raw)
                        elif 'dem' ==gridk:
                            ar = np.where(ar_raw<130, ar_raw, np.nan)
                            print(ar_raw.max())
                        else:
                            raise KeyError(gridk)
                            
                        #===========================================================
                        # #get styles by key
                        #===========================================================
                        if 'confuGrid' ==gridk:
                            cmap=confuGrid_cmap
                            norm=confuGrid_norm
                        elif gridk=='dem1':
                            cmap='plasma'
                            norm=None
                        elif ('dep' in gridk) or ('wd' in gridk):
                            cmap='viridis_r'
                            norm = matplotlib.colors.Normalize(vmin=0, vmax=4)
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
                        # asset samples-------
                        #=======================================================
                        if colk =='c2':# and 'pts_samples' in fp_lib[rowk]:
                            assert 'confuSamps' in fp_lib[rowk], f'{rowk} missing confuSamps'                                                        
 
                            #load
                            gdf = self._load_gdf(rowk, samples_fp=fp_lib[rowk]['confuSamps'])
                            
                            #drop Trues 
                            gdf1 = gdf.loc[~gdf['confusion'].isin(['TN', 'TP']), :]
                            
                            #map colors                            
                            gdf1['conf_color'] = gdf1['confusion'].replace(cc_d)                            
                            
                            #plot
                            _= gdf1.plot(column='conf_color', ax=ax, cmap=confuGrid_cmap, norm=confuGrid_norm,
                                     markersize=.2, marker='.', #alpha=0.8,
                                     )
                            
                            #pie chart                            
                            # Add a subplot to the lower right quadrant 
                            self._add_pie(ax, rowk, total_ser = gdf['confusion'].value_counts())
                        
 
                        #===========================================================
                        # post format
                        #===========================================================
                        #hide labels
                        ax.get_xaxis().set_ticks([])
                        ax.get_yaxis().set_ticks([])
                        
                        #add text
                        if gridk=='confuGrid' and isinstance(metric_lib, dict):
                            md = {k:v for k,v in metric_lib[rowk].items() if not k in cc_d.keys()}
                            #md = {**{rowk:''}, **md} 
                            ax.text(0.98, 0.05, get_dict_str(md), transform=ax.transAxes, 
                                    va='bottom', ha='right', fontsize=font_size, color='black',
                                    bbox=dict(boxstyle="round,pad=0.3", fc="white", lw=0.0,alpha=0.5 ),
                                    )
                            
                        #colorbar
                        if not gridk in axImg_d:
                            axImg_d[gridk]=[obj for obj in ax.get_children() if isinstance(obj, AxesImage)][0]
                            
                        
                        
                
                #===============================================================
                # pie plot--------
                #===============================================================
                elif gridk=='pie':
                    pass
 
 
        #=======================================================================
        # colorbar-------
        #=======================================================================
        for rowk, d0 in ax_d.items():
            for colk, ax in d0.items():
                #only last row 
                if not rowk==row_keys[-1]:
                    continue
                               
                gridk = grid_lib[rowk][colk]
                
 
                            
                if ('dep' in gridk) or ('wd' in gridk):
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
                    
                elif 'dem' in gridk:
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
                if colk=='c1':
                    pass
                    #ax.get_xscale()
                #turn off useless axis
                
                #first col
                if colk==col_keys[0]:
                    if rowk in rowLabels_d:
                        rowlab = rowLabels_d[rowk]
                    else:
                        rowlab = rowk
                        
                    ax.set_ylabel(rowlab)
 
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
        return self.output_fig(fig, ofp=ofp, logger=log, dpi=600, transparent=transparent)
    
    def _load_gdf(self, dkey, samples_fp=None, rmeta_d=None, confusion_codes=None):
        """convenienve to retrieve pre-loaded or load points"""
        #=======================================================================
        # defaults
        #=======================================================================
        if rmeta_d is None: rmeta_d=self.rmeta_d.copy()
        
        if samples_fp is None:
            samples_fp = self.fp_lib[dkey]['pts_samples']
            
        if confusion_codes is None:
            confusion_codes = self.confusion_codes.copy()
        
        #=======================================================================
        # preloaded
        #=======================================================================
        if dkey in self.gdf_d:
            gdf = self.gdf_d[dkey].copy()
            
        #=======================================================================
        # load
        #=======================================================================
        else:        
            gdf = gpd.read_file(samples_fp, bbox=rmeta_d['bounds']).rename(
                columns={'confusion':'code'})
            
            gdf['confusion'] = gdf['code'].replace({v:k for k,v in confusion_codes.items()})
 
        #=======================================================================
        # check
        #=======================================================================
        assert gdf.crs == rmeta_d['crs']
        
        return gdf.drop(['code', 'geometry'], axis=1).set_geometry(gdf.geometry)

    def _add_pie(self, ax, rowk,
                 total_ser=None,
                 font_size=None, 
                 confusion_color_d=None,
                 legend=True,
                 center_loc=(0.91, 0.2),
                 radius=0.075,
                 ):
         
        """add a pie chart to the axis
        
        Parameters
        --------
        radius: float
            radius of pie chart (relative ot the xdimensino)
        """
        #=======================================================================
        # defaults
        #=======================================================================
        if confusion_color_d is None:
            confusion_color_d=self.confusion_color_d.copy()
        
        if font_size is None:
            font_size=matplotlib.rcParams['font.size']
            
        #=======================================================================
        # #load data
        #=======================================================================
        if total_ser is None:
            gdf = self._load_gdf(rowk)
            total_ser = gdf['confusion'].value_counts() #.rename(nicknames_d2[rowk])
        
        colors_l = [confusion_color_d[k] for k in total_ser.index]
        
        #=======================================================================
        # get center location
        #=======================================================================
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        
        xdelta = xlim[1] - xlim[0]
        # Calculate the center of the pie chart, relative to the axis
        center_x = xlim[0] + center_loc[0] * (xdelta)
        center_y = ylim[0] + center_loc[1] * (ylim[1] - ylim[0])
 
        # ax.set_clip_box(ax.bbox)
        #=======================================================================
        # add pie
        #=======================================================================
    
        patches, texts = ax.pie(total_ser.values, colors=colors_l,
            # autopct='%1.1f%%',
            # pctdistance=1.2, #move percent labels out
            shadow=False,
            textprops=dict(size=font_size),
            # labels=total_ser.index.values,
            wedgeprops={"edgecolor":"black", 'linewidth':.5, 'linestyle':'solid', 'antialiased':True},
            radius=xdelta * radius,
            center=(center_x, center_y),
            frame=True)
        #=======================================================================
        # reset lims
        #=======================================================================
        ax.set_ylim(ylim)
        ax.set_xlim(xlim)
        #=======================================================================
        # legend
        #=======================================================================
        if legend:
            labels = [f'{k}:{v*100:1.1f}\%' for k, v in (total_ser / total_ser.sum()).to_dict().items()]
            ax.legend(patches, labels,
                      # loc='upper left',
                      loc='lower left',
                      ncols=len(labels),
                # bbox_to_anchor=(0, 0, 1.0, .1), 
                mode=None,  # alpha=0.9,
                frameon=True, framealpha=0.5, fancybox=False, alignment='left', columnspacing=0.5, handletextpad=0.2,
                fontsize=font_size - 2)
            
        return  

        
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

class Plot_hyd_HWMS(object):
    """plotting HWM error scatter for two simulations
    
    largely copied from ceLF.scrips.validate
    
    NOTE: this doesn't use any downscaling results
        so we could keep it separate
        decided to keep it integrated to avoid mismatch between specifying sim results 
            and the downscaling results pickles 
    
    """
    
    def __init__(self,
                 wd_key = 'water_depth', #key of validation ata
                 **kwargs):
        
        super().__init__(**kwargs)
        
        self.wd_key=wd_key
        
    def load_depth_samples(self, run_lib, hwm_fp, samp_fp=None, write=True, **kwargs):
        """pipeline loading of depth samples"""
        
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('load_dsamps', ext='.pkl', **kwargs)
        skwargs = dict(logger=log, tmp_dir=tmp_dir, out_dir=out_dir)
        
        #=======================================================================
        # load from source
        #=======================================================================
        if samp_fp is None:
            rlay_fp_lib = self.collect_hyd_fps(run_lib, **skwargs)
            
            gdf = self.get_depth_samples(rlay_fp_lib, hwm_fp=hwm_fp, **skwargs) #sample HWMs on depth rasters
            
            #write for dev
            if write:
                gdf.to_pickle(ofp)
                
                log.info(f'wrote {str(gdf.shape)} to \n    {ofp}')
            
        #=======================================================================
        # load precompiled
        #=======================================================================
        else:
            log.warning(f'loading precompiled depth samples from \n    {samp_fp}')
            gdf = pd.read_pickle(samp_fp)
 
        #=======================================================================
        # wrap
        #=======================================================================
        assert isinstance(gdf, gpd.geodataframe.GeoDataFrame)
        log.info(f'loaded {str(gdf.shape)}')
        
        return gdf
        
    def collect_hyd_fps(self, run_lib, **kwargs):
        """collect the filepaths from the run_lib"""
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('collect_rlay_fps', **kwargs)
        
        fp_lib={k:dict() for k in run_lib.keys()}
 
        
        #=======================================================================
        # pull for each
        #=======================================================================
        dem_fp, dep2=None, None
        for k0, d0 in run_lib.items(): #simulation name
            for k1, d1 in d0.items(): #cat0
                for k2, d2 in d1.items():
                    if k1=='smry':
                        if k2 in [
                            #'wse1', 'wse2', 
                            'dep2']:
                            if dep2 is None: dep2=d2 #name is misleading... this is just an input
                        elif k2 in ['dem1']:
                            dem_fp = d2                    
                        
                    elif k1=='vali':
 
                        if k2=='grid':
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
        fp_lib = {'dep1':dep1V, 
                  #'dem1':dem_fp, 
                  'dep2':dep2}
 
        log.info('got fp_lib:\n%s ' % (dstr(fp_lib) ))
        
        self.fp_lib = fp_lib
        return fp_lib 
    
    def plot_hyd_hwm(self,
                      gdf_raw,  
                      
                      ax=None,
                      figsize=(10*cm,10*cm), #(12, 9),
 
                      transparent=True,output_format=None,
                      font_size=None,
                      style_d = {
                          'dep1':dict(color='#996633', label='fine (s1)', marker='x'),
                          'dep2':dict(color='#9900cc', label='coarse (s2)', marker='o', fillstyle='none')
                                 },
                      
                      xlim = None,
 
                
                      **kwargs):
        """plot comparing performance of hyd models against HWMs
 
        rows: cols
            valid: 
            methods
        columns
            depthRaster r2, depthRaster r1, confusionRaster
        """
        #=======================================================================
        # defaults
        #=======================================================================
        if output_format is None: output_format=self.output_format
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('pHWM', ext='.'+output_format, **kwargs)

        if font_size is None:
            font_size=matplotlib.rcParams['font.size']
            
 
        assert isinstance(gdf_raw, pd.core.frame.DataFrame)
        
        #=======================================================================
        # data prep
        #=======================================================================
        #drop zeros
        bx = (gdf_raw==0).any(axis=1)
        log.info(f'dropping {bx.sum()}/{len(bx)}  samples with zeros')
        gdf = gdf_raw.loc[~bx, :]
        
        
        true_ser = gdf['obs']
        
        max_val = math.ceil(gdf.max().max())
        if xlim is None:
            xlim = (0, max_val)
        
        #=======================================================================
        # figure setup
        #=======================================================================
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            
        else:
            fig = ax.figure
        
        #=======================================================================
        # plot
        #=======================================================================
        meta_lib = dict()
        for k, ser in gdf.drop('obs', axis=1).items():
            
            #scatter
            xar, yar = true_ser.values, ser.values
            ax.plot(xar, yar ,linestyle='none', **style_d[k])
            
            #=======================================================================
            # compute error
            #=======================================================================
            slope, intercept, rvalue, pvalue, stderr =  scipy.stats.linregress(xar, yar)
            
            pearson, pval = scipy.stats.pearsonr(xar, yar)
            
            rmse = math.sqrt(np.square(xar - yar).mean())
            
            x_vals = np.array(xlim)
            y_vals = intercept + slope * x_vals
            
            d = dict(pearson=pearson, pval=pval, rvalue=rvalue, pvalue=pvalue, stderr=stderr, rmse=rmse)
            meta_lib[k] = d
            log.info(f'for {k} got \n    {d}')
            
            #===================================================================
            # plot correlation
            #===================================================================
            ax.plot(x_vals, y_vals, color=style_d[k]['color'], linewidth=0.5)
            
        
        #1:1 line
        
        ax.plot([0.0, max_val*1.1], [0.0, max_val*1.1], color='black', label='1:1', linewidth=0.5, linestyle='dashed')
            
        
        #=======================================================================
        # post
        #=======================================================================
        #make square
        
        ax.set_ylim(0, max_val)
        ax.set_xlim(0, max_val)
        ax.set_aspect('equal', adjustable='box')
        
        ax.set_xlabel('observed HWM (m)')
        ax.set_ylabel('simulated max depth (m)')
        
        ax.legend()
        
        return self.output_fig(fig, logger=log, ofp=ofp, transparent=transparent)
    
        plt.show()
 
    
    def get_hwm(self, fp, **kwargs):
        """load hwm data"""
        assert os.path.exists(fp), f'bad path for hwm: {fp}'
        
        gdf = gpd.read_file(fp, **kwargs)
        
        assert len(gdf) > 0
        assert np.all(gdf.geometry.geom_type == 'Point')
        
        assert self.wd_key in gdf, self.wd_key
        
        return gdf
    
    def get_hwm_pts(self, gdf=None, hwm_fp=None):
        """get the points from the HWM data"""
        #load
        if gdf is None:
            gdf = self.get_hwm(hwm_fp)

        return gdf.geometry
    
    def get_simulated(self, samp_gs, rlay_fp,
                      colName=None,
                      **kwargs):
        """retrieve depths from simulation
        
        Parameters
        -----------
        samp_gs: GeoSeries
            points to sample
            
        colName: str
            what to name the new series
            
        """
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('get_sim', **kwargs)
        
        if colName is None:
            colName = os.path.basename(rlay_fp) 
 
        #=======================================================================
        # get raster
        #=======================================================================
            
        log.info(f'loading depth grid from {os.path.basename(rlay_fp)}')
        
        ds = rio.open(rlay_fp, mode='r')
        
        #=======================================================================
        # sample points
        #=======================================================================  
            
        samp_gdf_raw = get_samples(samp_gs, ds, colName=colName)
        
        samp_gdf = samp_gdf_raw.dropna(axis=0, subset=colName)
        
        stats_d = {n:getattr(samp_gdf[colName], n)() for n in ['mean', 'min', 'max']}
        
        log.info(f'got {len(samp_gdf)} samples on {colName}\n    {stats_d}')
        
        assert isinstance(samp_gdf, gpd.geodataframe.GeoDataFrame)
 
        return samp_gdf
    
    def get_depth_samples(self, rlay_fp_lib, hwm_fp=None,  hwm_gdf=None, **kwargs):
        """wrapper for loading and building errosr"""
        
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('get_depth_samples', **kwargs)
        skwargs = dict(logger=log, tmp_dir=tmp_dir, out_dir=out_dir)
        #=======================================================================
        # load the data
        #=======================================================================
        if hwm_gdf is None:
            log.info(f'loading HWMs from {hwm_fp}')
            
            #get bbox
            rlay_meta_d = get_meta(rlay_fp_lib['dep1'])
            
            
            hwm_gdf = self.get_hwm(hwm_fp, bbox=rlay_meta_d['bounds'])
        else:
            assert hwm_fp is None
        
        #=======================================================================
        # sample the simulated raster
        #=======================================================================
        d=dict()
        for k, rlay_fp in rlay_fp_lib.items():
            samp_gdf = self.get_simulated(hwm_gdf.geometry, rlay_fp, colName=k, **skwargs)
            
            d[k] = samp_gdf[k]
        
        d['obs'] = hwm_gdf[self.wd_key] #add HWMs
        #=======================================================================
        # wrap
        #=======================================================================
        res_gdf = pd.concat(d, axis=1).set_geometry(hwm_gdf.geometry)
 
        log.info(f'assembled obs and sim w/ {str(res_gdf.shape)}')
        return res_gdf
 
    def get_err(self, gdf, logger=None, **kwargs):
        if logger is None: logger=self.logger
        with ErrorCalcs(pred_ser=gdf['sim'], true_ser=gdf['obs'],logger=logger, **kwargs) as wrkr:
            err_d = {
                'count':len(gdf), 
                'rmse':wrkr.get_RMSE(), 
                'meanError':wrkr.get_meanError(), 
                'bias':wrkr.get_bias()}
            confusion_df, confusion_dx = wrkr.get_confusion(wetdry=True)
        return confusion_df, err_d, confusion_dx

 
 

class PostSession(Plot_rlays_wrkr, Plot_samples_wrkr, Plot_hyd_HWMS,
                  Plotr, ValidateSession):
    sim_color_d = {'vali':'black', 'nodp':'orange', 'cgs':'teal', 'bgl':'violet', 's14':'#24855d'}
    confusion_color_d = {
            'FN':'#c700fe', 'FP':'red', 'TP':'#00fe19', 'TN':'white'
            }
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
        
        #print contents
        k1=list(run_lib.keys())[0]
        print(f'\nfor {k1}')
        print(pprint.pformat(run_lib[k1], width=30, indent=0.3, compact=True, sort_dicts =False))
        
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
        
def basic_post_pipeline(meta_fp_d, 
                      sample_dx_fp=None,
 
                      rlay_mat_kwargs= dict(),
                      samples_mat_kwargs=dict(),
                      hyd_hwm_kwargs=dict(),
                      **kwargs):    
    
    res_d = dict()
    with PostSession(**kwargs) as ses:
        
        #load the metadata from teh run
        run_lib, smry_d = ses.load_metas(meta_fp_d)
        
        
        #=======================================================================
        # hydrodyn HWM performance
        #=======================================================================
        #=======================================================================
        # gdf = ses.load_depth_samples(run_lib, hwm_fp, 
        #                              samp_fp = r'L:\10_IO\fdsc\outs\ahr_aoi08_0130\post_0206\20230302\ahr_aoi08_0130_post_0206_0302_load_dsamps.pkl'
        #                              )  
        #  
        # ses.plot_hyd_hwm(gdf.drop('geometry', axis=1), **hyd_hwm_kwargs)
        #=======================================================================
  
        
        #=======================================================================
        # RASTER PLOTS
        #=======================================================================
        #get rlays
        rlay_fp_lib, metric_lib = ses.collect_rlay_fps(run_lib)
         
        #plot them
        res_d['rlay_mat'] = ses.plot_rlay_mat(rlay_fp_lib, metric_lib, **rlay_mat_kwargs)
         
 
        return

 
        #=======================================================================
        # sample metrics
        #=======================================================================
        ses.logger.info('\n\nSAMPLES\n\n')
        try: 
            del run_lib['nodp'] #clear this
        except: pass
        
        df, metric_lib = ses.collect_samples_data(run_lib, sample_dx_fp=sample_dx_fp)
        
        """switched to plotting all trues per simulation"""
 
        df_wet=df

        res_d['ssampl_mat'] =ses.plot_samples_mat(df_wet, metric_lib, **samples_mat_kwargs)
        
    print('finished on \n    ' + pprint.pformat(res_d, width=30, indent=True, compact=True, sort_dicts =False))
    return res_d
 
