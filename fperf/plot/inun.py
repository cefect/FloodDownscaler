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
import rasterio as rio
import geopandas as gpd
import scipy

from rasterio.plot import show

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.image import AxesImage
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

 

from hp.plot import Plotr, get_dict_str, hide_text, cm
from hp.pd import view
from hp.rio import (    
    get_ds_attr,get_meta, SpatialBBOXWrkr
    )
from hp.err_calc import get_confusion_cat, ErrorCalcs
from hp.gpd import get_samples
from hp.fiona import get_bbox_and_crs

 
from fperf.plot.grids import Plot_grids


class Plot_inun_peformance(Plot_grids):
    """worker for plotting inundation performance of downscaling and hydro methods"""
    gdf_d=dict() #container for preloading geopandas
    
    metric_labels = {'hitRate': 'Hit Rate', 
                     'falseAlarms':'False Alarms', 
                     'criticalSuccessIndex':'Crit. Suc. Index', 
                     'errorBias':'Error Bias',
                     }
    
 
    def plot_inun_perf_mat(self,
                      fp_df, metric_lib=None,
 
                      row_keys=None,col_keys = None, 
 
                      confusion_color_d=None,
                      output_format=None,
                      rowLabels_d = None,
                      #pie_legend=True, 
                      box_fp=None, 
                      arrow_kwargs_lib=dict(),
                      fig_mat_kwargs=dict(figsize=None),
                      **kwargs):
        """matrix plot comparing methods for downscaling: rasters
 
        rows: simNames
        columns
            depth grid (w/ asset exposure)
            confusion grid (w/ inundation metrics)
            
        Pars
        --------
        fp_lib: dict
            {row key i.e. method name: {gridName: filepath}}
            
        box_fp: str
            optional filepath to add a black focus box to the plot
        """
        #=======================================================================
        # defaults
        #=======================================================================
        if output_format is None: output_format=self.output_format
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('inunPerf', ext='.'+output_format, **kwargs)
        
        log.info(f'on {str(fp_df.shape)}')
 
        font_size=matplotlib.rcParams['font.size']
        if confusion_color_d is None:
            confusion_color_d=self.confusion_color_d.copy()
            
        if rowLabels_d is None:
            rowLabels_d=self.rowLabels_d
        if rowLabels_d is None:
            rowLabels_d=dict()
        
        #get confusion style shortcuts
        cc_d = self.confusion_codes.copy()
        cval_d = self.confusion_val_d.copy()
        
        #spatial meta from dem for working with points
        rmeta_d = get_meta(fp_df.iloc[0,0])
         
        
        #bounding box
        if box_fp is None:
            focus_bbox=None
        else:
            
            focus_bbox, crs=get_bbox_and_crs(box_fp)
            log.info(f'using aoi from \'{os.path.basename(box_fp)}\'')
            assert crs == rmeta_d['crs']
 
        #=======================================================================
        # setup figure
        #=======================================================================
        if row_keys is None:
            row_keys = fp_df.index.to_list()
 
            
        assert set(row_keys).difference(fp_df.index.values)==set()
            
        if col_keys is None:
            col_keys = fp_df.columns.to_list()
            
            
        assert set(col_keys).difference(fp_df.columns.values)==set()
 
 
        fig, ax_d = self.get_matrix_fig(row_keys, col_keys, logger=log, 
                                        set_ax_title=False, constrained_layout=True,
                                        **fig_mat_kwargs)
 
        #=======================================================================
        # plot loop------
        #=======================================================================
        focus_poly = None
        axImg_d = dict() #container for objects for colorbar
        #dep1_yet=False
        for rowk, d0 in ax_d.items():
            for colk, ax in d0.items():                
                gridk = colk.upper()
                #aname = nicknames_d2[rowk]
                aname=rowLabels_d[rowk]                
                log.debug(f'plot loop for {rowk}.{colk}.{gridk} ({aname})')
                
 
                #===============================================================
                # raster plot-----
                #===============================================================
 
                fp = fp_df.loc[rowk, colk]
                
                log.info(f'plotting {rowk} x {colk} ({gridk}): {os.path.basename(fp)}')
                
                self._ax_raster_show(ax,fp, gridk=gridk)                    
 
                    
                #=======================================================
                # asset samples (pie charts)-------
                #=======================================================
                """this made more sense when we were taking simulated grids as true""" 
                #===============================================================
                # if gridk =='WSH':# and 'pts_samples' in fp_lib[rowk]:
                #     assert 'confuSamps' in fp_lib[rowk], f'{rowk} missing confuSamps'                                                        
                #  
                #     #load
                #     gdf = self._load_gdf(rowk, samples_fp=fp_lib[rowk]['confuSamps'], rmeta_d=rmeta_d)
                #      
                #     #drop Trues 
                #     gdf1 = gdf.loc[~gdf['confusion'].isin(['TN', 'TP']), :]
                #      
                #     #map colors                            
                #     gdf1['conf_color'] = gdf1['confusion'].replace(cc_d)                            
                #      
                #     #plot
                #     _= gdf1.plot(column='conf_color', ax=ax, cmap=confuGrid_cmap, norm=confuGrid_norm,
                #              markersize=.2, marker='.', #alpha=0.8,
                #              )
                #      
                #     #pie chart                            
                #     # Add a subplot to the lower right quadrant 
                #     self._add_pie(ax, rowk, total_ser = gdf['confusion'].value_counts(), legend=pie_legend)
                #===============================================================
                    
                #===========================================================
                # focus box--------
                #===========================================================
                #if (colk=='c3') and (not focus_bbox is None) and (rowk==row_keys[0]): #first map only
                if colk==col_keys[-1] and rowk==row_keys[0] and not focus_bbox is None:
                    x, y = focus_bbox.exterior.coords.xy 
                    polygon_points = [[x[i], y[i]] for i in range(len(x))]                        
                                                        
                    focus_poly=ax.add_patch(
                        matplotlib.patches.Polygon(polygon_points, edgecolor='blue', facecolor='none', linewidth=1.0, linestyle='dashed')
                        )
                    
                    ax.legend([focus_poly], ['detail'], loc='upper right')
 
                #===========================================================
                # post format-------
                #===========================================================
                #hide labels
                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])
                
                #add text
                if gridk=='CONFU' and isinstance(metric_lib, dict):
                    md = {k:v for k,v in metric_lib[rowk].items() if not k in cc_d.keys()}
                    """
                    md.keys()
                    """
                    #clean names
                    for k,v in md.copy().items():
                        if k in self.metric_labels:
                            md.pop(k)
                            md[self.metric_labels[k]]=v
                        
 
 
 
                    ax.text(0.98, 0.05, get_dict_str(md, num_format = '{:.3f}'), transform=ax.transAxes, 
                            va='bottom', ha='right', fontsize=font_size, color='black',
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", lw=0.0,alpha=0.8),
                            )
                else:
                    self._add_scaleBar_northArrow(ax)
                    
                # colorbar
                if not gridk in axImg_d:
                    axImg_d[gridk] = [obj for obj in ax.get_children() if isinstance(obj, AxesImage)][0]
                
 
        #=======================================================================
        # colorbar-------
        #=======================================================================
        for rowk, d0 in ax_d.items():
            for colk, ax in d0.items():
                #only last row 
                if not rowk==row_keys[-1]:
                    continue
                               
                gridk = colk
                            
                location, fmt, label, spacing = self._get_colorbar_pars_by_key(gridk)
                
                cbar = fig.colorbar(axImg_d[gridk],
                                #cax=cax, 
 
                                orientation='horizontal',
                         ax=ax, location=location, # steal space from here
                         extend='both', #pointed ends
                         format=fmt, label=label,spacing=spacing,
                         shrink=0.8,
                         )
                
                #relabel
                if 'CONFU' == gridk: 
                    cbar.set_ticks([(101-1)/2+1, 101.5, (111-102)/2+102, 111.5], 
                                   labels = [{v:k for k,v in cc_d.items()}[k0] for k0 in cval_d.keys()] 
                                   )
                    
        #=======================================================================
        # add annotation arrow------
        #=======================================================================
        if len(arrow_kwargs_lib)>0:
            for k, arrow_kwargs in arrow_kwargs_lib.items():
                self._add_arrow(ax_d, logger=log, **arrow_kwargs)
                
                """
                plt.show()
                """
            
        #=======================================================================
        # post format-------
        #======================================================================= 
        for rowk, d0 in ax_d.items():
            for colk, ax in d0.items():
 
                
                #first col
                if colk==col_keys[0]:
                    if rowk in rowLabels_d:
                        rowlab = rowLabels_d[rowk]
                    else:
                        rowlab = rowk
                        
                    ax.set_ylabel(rowlab)
 
 
        #=======================================================================
        # wrap
        #=======================================================================
        log.info('finished')
        return self.output_fig(fig, ofp=ofp, logger=log, dpi=600)
    
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

