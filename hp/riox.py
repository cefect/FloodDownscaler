'''
Created on Sep. 23, 2022

@author: cefect
'''
import os
import numpy as np
import matplotlib.pyplot as plt
from hp.basic import get_dict_str


def plot_rast(ar_raw,
              ax=None,
              cmap='gray',
              interpolation='nearest',
              txt_d = None,
 
              transform=None,
              **kwargs):
    """plot a raster array
    see also hp.plot
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
    #ax_img = show(ar_raw, transform=transform, ax=ax,contour=False, **imkwargs)
    ax_img = ax.imshow(masked_ar,cmap=cmap,interpolation=interpolation, **kwargs)
 
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