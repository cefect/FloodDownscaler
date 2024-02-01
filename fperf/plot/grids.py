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
 

 
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.image import AxesImage
 


 
from hp.pd import view
from hp.rio import (    
    get_ds_attr,get_meta, SpatialBBOXWrkr, get_bbox, get_data_stats
    )
 
from hp.fiona import get_bbox_and_crs

from fperf.plot.base import PostBase


class Plot_grids(PostBase):
    """worker for plotting raw raster  results"""
 
    def plot_grids_mat(self,
                          fp_d,
                          gridk=None, #for formatting by grid type
                          mod_keys=None,
                          
                          dem_fp=None,
                          inun_fp=None,
                          aoi_fp=None,
                          
                          output_format=None,rowLabels_d=None,add_subfigLabel=None,
                          colorBar_bottom_height=0.15,
 
                          vmin=None, vmax=None,
                          show_kwargs=None,
                          fig_mat_kwargs=dict(),
                          arrow_kwargs_lib=dict(),
                          inun_kwargs = dict(facecolor='none', 
                                             edgecolor='black', 
                                             linewidth=0.75, 
                                             linestyle='dashed'),
                          **kwargs):
        """matrix plot of raster results. nice for showing a small region of multiple sims
        
        Pars
        --------
        fp_lib: dict
            filepaths of grids for plotting
                {modk:{gridk ('dem', 'true_inun', ...):fp}}
        gridk: str, default: 'pred_wse'
            which grid to plot
            
        add_subfigLabel: bool
            True: add journal sub-figure labelling (a0)
            False: use the fancy labels
            
        show_kwargs: dict
            over-ride default grid show kwargs (should be a lvl2 dict?)
            
        """
        #=======================================================================
        # defaults
        #=======================================================================
        if output_format is None: output_format=self.output_format
        if gridk is None:
            dkey = 'grids'
        else:
            dkey = f'grids{gridk.upper()}'
        log, tmp_dir, out_dir, ofp, resname = self._func_setup(dkey, ext='.'+output_format, **kwargs)
        
        
        #subfigure labelling
        """either add journal labelling (a0, etc.) or fancy label"""
        if add_subfigLabel is None:
            if 'add_subfigLabel' in  fig_mat_kwargs:
                add_subfigLabel = fig_mat_kwargs.pop('add_subfigLabel')
            else:
                add_subfigLabel=self.add_subfigLabel
        
        fig_mat_kwargs['add_subfigLabel'] = add_subfigLabel
            
 
        
        #list of model values
        if mod_keys is None:
            mod_keys = list(fp_d.keys())
            #mod_keys = ['WSE2', 'Basic', 'SimpleFilter', 'Schumann14', 'CostGrow','WSE1']            
        assert set(mod_keys).difference(fp_d.keys())==set()
        
        #model fancy labels
        if rowLabels_d is None:
            rowLabels_d=self.rowLabels_d
            
        if rowLabels_d is None:
            rowLabels_d = dict()
            
        #add any missing
        for k in mod_keys:
            if not k in rowLabels_d:
                rowLabels_d[k] = k
            
        #bounding box
        rmeta_d = get_meta(dem_fp)  #spatial meta from dem for working with points
        
        if aoi_fp is None:
            bbox, crs=get_bbox(dem_fp), self.crs
        else:            
            bbox, crs=get_bbox_and_crs(aoi_fp)
            log.info(f'using aoi from \'{os.path.basename(aoi_fp)}\'')
            
        
        
               
        assert crs.to_epsg()==rmeta_d['crs'].to_epsg()
            
        log.info(f'plotting {len(fp_d)} on {mod_keys}')
        #=======================================================================
        # setup figure
        #=======================================================================        
        ax_d, mat_df, row_keys, col_keys, fig = self._get_fig_mat_models(
                                            mod_keys,logger=log, 
                                            constrained_layout=False, 
                                            **fig_mat_kwargs)
 
        
        #=======================================================================
        # plot loop------
        #=======================================================================        
        meta_lib, axImg_d=dict(), dict()
        for rowk, d0 in ax_d.items():
            for colk, ax in d0.items():                
                #===============================================================
                # setup
                #===============================================================
                modk = mat_df.loc[rowk, colk]
                if modk=='nan': 
                    ax.axis('off')
                    continue              
                log.info(f'plotting {rowk}x{colk} ({modk})\n    {fp_d[modk]}')
                
                #===============================================================
                # plot it
                #===============================================================
                # DEM raster 
                self._ax_raster_show(ax,  dem_fp, bbox=bbox,gridk='hillshade')
                
                # focal raster
                fp = fp_d[modk]
                self._ax_raster_show(ax,  fp, bbox=bbox,gridk=gridk, alpha=0.9, show_kwargs=show_kwargs,
                                     vmin=vmin, vmax=vmax)
                
                log.debug(get_data_stats(fp))
                #inundation                
                gdf = gpd.read_file(inun_fp)
                assert gdf.geometry.crs==crs, f'crs mismatch: {gdf.geometry.crs}\n    {inun_fp}'
 
                #boundary 
                gdf.clip(bbox.bounds).plot(ax=ax,**inun_kwargs)
 
                #===============================================================
                # label
                #=============================================================== 
                ax.text(0.95, 0.05, 
                            rowLabels_d[modk], 
                            transform=ax.transAxes, va='bottom', ha='right',
                            size=matplotlib.rcParams['axes.titlesize'],
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", lw=0.0,alpha=0.5 ),
                            )
                
                #===============================================================
                # #scale bar
                #===============================================================
                self._add_scaleBar_northArrow(ax)
                    
                #===============================================================
                # wrap
                #===============================================================
 
                # hide labels
                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])
                
        #=======================================================================
        # add annotation arrows------
        #=======================================================================
        """
        plt.show()
        """
        if len(arrow_kwargs_lib)>0:
            for k, arrow_kwargs in arrow_kwargs_lib.items():
                self._add_arrow(ax_d, logger=log, **arrow_kwargs)
        #=======================================================================
        # colorbar-------
        #=======================================================================
        #grab the image object for making the colorbar
        ax = ax_d[row_keys[0]][col_keys[0]] #use the first axis
        l= [obj for obj in ax.get_children() if isinstance(obj, AxesImage)]
        axImg_d = dict(zip(['dem', gridk], l))
                
        log.debug(f'adding colorbar')
        
        #get parameters
        _, fmt, label, spacing = self._get_colorbar_pars_by_key(gridk)
        shared_kwargs = dict(orientation='horizontal',
                             extend='both', #pointed ends
                             shrink=0.8,
                             ticklocation='top',
                             )
        
        #add the new axis
        fig.subplots_adjust(bottom=colorBar_bottom_height, 
                            wspace=0.05, top=0.999, hspace=0.05, left=0.05, right=0.95) 
        
        cax = fig.add_axes((0.1, 0.01, 0.8, 0.03)) #left, bottom, width, height
        
        #add the bar
        cbar = fig.colorbar(axImg_d[gridk],cax=cax,label=label,format=fmt, spacing=spacing,
                                 **shared_kwargs)
        
        
        
        #=======================================================================
        # post
        #=======================================================================
        """nothing in the legend for some reason...."""
        for rowk, d0 in ax_d.items():
            for colk, ax in d0.items():
                 
                if rowk==row_keys[0]:
                    if colk==col_keys[-1]:
                        dummy_patch = matplotlib.patches.Patch(label='observed', **inun_kwargs)
                        ax.legend(handles=[dummy_patch], loc='upper right')
                        

 
                        
 
 
        
        return ax_d, fig
        
 
        


 
