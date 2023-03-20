'''
Created on Mar. 20, 2023

@author: cefect
'''
import os, string
import numpy as np
import rasterio as rio
from rasterio.plot import show


import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex

import earthpy.spatial as es

from hp.plot import Plotr, get_dict_str
from hp.rio import confusion_codes

class RioPlotr(Plotr):
    """grid plotting"""
    
    
    #standard styles for grids
    grid_styles_lib={
        'hillshade':dict(cmap=plt.cm.copper, norm=None, alpha=0.8),
        'dem':dict(cmap = 'plasma', norm = None),
        'wsh':dict(cmap = 'viridis_r', norm = matplotlib.colors.Normalize(vmin=0, vmax=4)),
        'wse':dict(cmap = 'plasma_r', norm = None), #May lead to inconsistent color styles
        }
    
    #confusion colors
    #https://colorbrewer2.org/?type=diverging&scheme=RdYlBu&n=4
    confusion_color_d = {
            'FN':'#d7191c', 'FP':'#fdae61', 'TP':'#abd9e9', 'TN':'#2c7bb6'
            }
    
    def __init__(self, **kwargs):
        
        super().__init__(**kwargs)
        self.confusion_codes=confusion_codes
        self._set_confusion_style()
        
    
    #===========================================================================
    # style
    #===========================================================================
    def _set_confusion_style(self):
        """setup styles for confusion plotting"""
        
        confusion_color_d=self.confusion_color_d.copy()
        cc_d = self.confusion_codes.copy()
        
        #get rastetr val to color conversion for confusion grid
        cval_d = {v:confusion_color_d[k] for k,v in cc_d.items()}        
        cval_d = {k:cval_d[k] for k in sorted(cval_d)} #sort it
        
        cmap = matplotlib.colors.ListedColormap(cval_d.values())        
        norm = matplotlib.colors.BoundaryNorm(
                                    np.array([0]+list(cval_d.keys()))+1, #bounds tt capture the data 
                                      ncolors=len(cval_d),
                                      #cmap.N, 
                                      extend='neither',
                                      clip=True,
                                      )
        
        self.grid_styles_lib['confusion'] = {'cmap':cmap, 'norm':norm}
        
    
    def _mask_grid_by_key(self, ar_raw, gridk, cc_d={'TN':100}):
        """apply a mask to the grid based on the grid type"""
        if gridk=='wsh':
            assert np.any(ar_raw == 0), 'depth grid has no zeros '
            ar = np.where(ar_raw == 0, np.nan, ar_raw)
        elif 'confuGrid' in gridk:
            # mask out true negatives
            ar = np.where(ar_raw == cc_d['TN'], np.nan, ar_raw)
        elif 'dem' == gridk:
            ar = np.where(ar_raw < 130, ar_raw, np.nan)
 
        elif 'wse' in gridk: #no masking
            ar = ar_raw
        elif 'hillshade'==gridk:
            ar = es.hillshade(ar_raw)
        else:
            raise KeyError(gridk)
        return ar
    
    def _get_colorbar_pars_by_key(self, gridk):
        """get standard colorbar parameters based on the grid type"""
        if gridk=='wsh':
            spacing = 'proportional'
            label = 'WSH (m)'
            fmt = matplotlib.ticker.FuncFormatter(lambda x, p:'%.1f' % x)
            location = 'bottom'
        elif 'confuGrid' in gridk:
            #spacing='proportional'
            spacing = 'uniform'
            label = 'Confusion'
            fmt = None
            #fmt = matplotlib.ticker.FuncFormatter(lambda x, p:cc_di[x])
            #cax=cax_bot
            location = 'bottom'
        elif 'dem' in gridk:
            spacing = 'proportional'
            label = 'DEM (masl)'
            fmt = matplotlib.ticker.FuncFormatter(lambda x, p:'%.0f' % x)
            location = 'bottom'
 
            
        elif 'wse' in gridk:
            spacing = 'proportional'
            fmt = matplotlib.ticker.FuncFormatter(lambda x, p:'%.1f' % x)
            label = 'WSE (masl)'
            location='bottom'
            
 
        else:
            raise KeyError( gridk)
            
        return location, fmt, label, spacing
        
    #===========================================================================
    # plotters------
    #===========================================================================
    def _ax_raster_show(self, ax, bbox, fp, 
                        gridk=None,
                        show_kwargs=None,
                         **kwargs):
        """add a styleized raster to the axis"""
        #=======================================================================
        # defaults
        #=======================================================================
 
        with rio.open(fp, mode='r') as ds:
            
            #===================================================================
            # #load and clip the array
            #===================================================================
            if bbox is None:
                window = None
                transform = ds.transform
            else:
                window = rio.windows.from_bounds(*bbox.bounds, transform=ds.transform)
                #transform = rio.transform.from_bounds(*bbox.bounds, *window.shape)
                transform = rio.windows.transform(window, ds.transform)
                
            ar_raw = ds.read(1, window=window, masked=True)
            #===========================================================
            # #apply masks
            #===========================================================
            ar = self._mask_grid_by_key(ar_raw, gridk)
            #===========================================================
            # #get styles by key
            #===========================================================
            if show_kwargs is None:
                if gridk is None:
                    show_kwargs=dict() 
                else:
                    assert gridk in self.grid_styles_lib, f'no gridk \'{gridk}\' found in style lib'
                    show_kwargs = self.grid_styles_lib[gridk]
                    
            assert isinstance(show_kwargs, dict)
        #===========================================================
        # plot it
        #===========================================================
        """
        plt.show()
        """
 
        return show(ar, 
                    transform=transform, 
                    ax=ax, contour=False,interpolation='nearest',**show_kwargs, **kwargs)
        
#===============================================================================
# FUNCTIONS------
#===============================================================================
def plot_rast(ar_raw,
              ax=None,
              cmap='gray',
              interpolation='nearest',
              txt_d = None,
 
              transform=None,
              **kwargs):
    """plot a raster array
 
    TODO: add a histogram"""
    
    #===========================================================================
    # defaults
    #===========================================================================
    if ax is None:
        fig, ax = plt.subplots()  # Create a figure containing a single axes.
        limits = None
    else:
        limits = ax.axis()
        
    if txt_d is None: txt_d=dict()
    
    imkwargs = {**dict(cmap=cmap,interpolation=interpolation), **kwargs}
    
    #===========================================================================
    # plot the image
    #===========================================================================
    ax_img = show(ar_raw, transform=transform, ax=ax,contour=False, **imkwargs)
    #ax_img = ax.imshow(masked_ar,cmap=cmap,interpolation=interpolation, **kwargs)
 
    #plt.colorbar(ax_img, ax=ax) #steal some space and add a color bar
    #===========================================================================
    # add some details
    #===========================================================================
    txt_d.update({'shape':str(ar_raw.shape), 'size':ar_raw.size})
 
    ax.text(0.1, 0.9, get_dict_str(txt_d), transform=ax.transAxes, va='top', fontsize=8, color='red')
    
    #===========================================================================
    # wrap
    #===========================================================================
    if not limits is None:
        ax.axis(limits)
    """
    plt.show()
    """
    
    return ax

def plot_rast2(ar_raw,
              ax=None,
              cmap='gray',
              interpolation='nearest',
              txt_d = None,
              nodata=None,
              **kwargs):
    """plot a raster array
    
 
    
    TODO: add a histogram"""
    #===========================================================================
    # defaults
    #===========================================================================
    if ax is None:
        fig, ax = plt.subplots()  # Create a figure containing a single axes.
        
    if txt_d is None: txt_d=dict()
    
    #===========================================================================
    # handle nodata
    #===========================================================================
    if not nodata is None:
        masked_ar = np.ma.masked_where(ar_raw==nodata, ar_raw)
        
        #update meta
        txt_d.update({'nodata':'%.4e'%nodata, 'ndcnt':masked_ar.mask.sum()})
 
    else:
        masked_ar=ar_raw
        txt_d['nodata']='none'
    
    #===========================================================================
    # plot the image
    #===========================================================================
    
    ax_img = ax.imshow(masked_ar,cmap=cmap,interpolation=interpolation, **kwargs)
 
    #plt.colorbar(ax_img, ax=ax) #steal some space and add a color bar
    #===========================================================================
    # add some details
    #===========================================================================
    txt_d.update({'shape':str(ar_raw.shape), 'size':ar_raw.size})
 
    ax.text(0.1, 0.9, get_dict_str(txt_d), transform=ax.transAxes, va='top', fontsize=8, color='red')
    """
    plt.show()
    """
    
    return ax
    