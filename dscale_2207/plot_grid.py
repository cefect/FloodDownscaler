'''
Created on Jan. 9, 2023

@author: cefect

plot a raster as a table/grid
'''
import os, logging, sys
import numpy as np
import rasterio as rio
from rasterio.plot import show
 

from hp.plot import Plotr
from hp.oop import Session

#===============================================================================
# setup matplotlib----------
#===============================================================================
cm = 1/2.54

output_format='png'
usetex=False
if usetex:
    os.environ['PATH'] += R";C:\Users\cefect\AppData\Local\Programs\MiKTeX\miktex\bin\x64"
    
  
import matplotlib
#matplotlib.use('Qt5Agg') #sets the backend (case sensitive)
matplotlib.set_loglevel("info") #reduce logging level
import matplotlib.pyplot as plt
 
#set teh styles
plt.style.use('default')
 
#font
matplotlib.rc('font', **{
        'family' : 'serif',
        'weight' : 'normal',
        'size'   : 8})
 
 
for k,v in {
    'axes.titlesize':10,
    'axes.labelsize':10,
    'xtick.labelsize':8,
    'ytick.labelsize':8,
    'figure.titlesize':12,
    'figure.autolayout':False,
    'figure.figsize':(10,10),
    'legend.title_fontsize':'large',
    'text.usetex':usetex,
    }.items():
        matplotlib.rcParams[k] = v
  
print('loaded matplotlib %s' % matplotlib.__version__)
    


class Plot_RasterToTable(Plotr, Session):
    def plot_set(self, pars_d, **kwargs):
        """plot a set of rasters"""
        #=======================================================================
        # defaults
        #=======================================================================
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('Plot_RasterToTable',ext='.png', **kwargs)
        
        res_d=dict()
        
        #=======================================================================
        # loop and plot
        #=======================================================================
        log.info(f'plotting {len(pars_d)} rasters')
        for name, pars_d in pars_d.items():
            fp = pars_d.pop('fp')
            log.info(f'plotting {name} w/\n    {pars_d}')
            res_d[name] = self.plot_rasterToTable(fp, resname = self._get_resname(dkey=f'rasterTable_{name}'),
                                             logger=log.getChild(name), **pars_d)
            
        return res_d
        
    
    def plot_rasterToTable(self, fp, 
                           cmap='virids', norm=None,
                           **kwargs):
        """main caller to plot the raster file"""
        assert os.path.exists(fp), fp
        
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('Plot_RasterToTable',ext='.svg', **kwargs)
        
        log.debug(f'from {fp}')
        #=======================================================================
        # setup plot
        #=======================================================================
        fig, ax = plt.subplots(tight_layout=False)
        
        #=======================================================================
        # load raster
        #=======================================================================
        with rio.open(fp, mode='r') as ds:
            ar_raw = ds.read(1, window=None, masked=True)
            log.info(f'plotting raster on {ar_raw.shape}')
            #mask
            
        #===========================================================
        # #raster plot
        #===========================================================
        #_ = show(ar,transform=ds.transform,ax=ax,contour=False, cmap=cmap, interpolation='nearest',norm=norm)
        
        ax_img = ax.imshow(ar_raw, cmap=cmap, norm=norm, aspect='equal', zorder=1, alpha=0.5)
        
        #===================================================================
        # table plot
        #===================================================================
        #round values
        ar=ar_raw.round(1)
        #===================================================================
        # bbox_obj = ax.get_tightbbox(for_layout_only=True)
        # bbox = [bbox_obj.x0, bbox_obj.y0, bbox_obj.width, bbox_obj.height]
        #===================================================================
        #transparent backgrounds
        cell_colors = [[matplotlib.colors.to_rgba('white', alpha=0.0) for j in range(ar.shape[1])] for i in range(ar.shape[0])]
        bbox = [0, 0, 1, 1]
        
        table = ax.table(cellText=ar,  loc='center', bbox=bbox, zorder=2, cellColours=cell_colors, cellLoc='center',
                 #fontsize=32,
                 )
        
        table.set_fontsize(16)
        
        #===================================================================
        # format
        #===================================================================
        #turn off spines and tickmarks
        ax.axis('off') 
        
        fig.tight_layout()
        #=======================================================================
        # wrap
        #=======================================================================
        
        return self.output_fig(fig, ofp=ofp, transparent=False, dpi=300, bbox_inches='tight', add_stamp=False)
            
 
            
            
        
    
def run01(pars_d = {
        'final':dict(
            fp=r'l:\10_IO\2207_dscale\outs\fdsc\toy\200230205\none\FloodDownscaler_test_0205_dsc.tif',
              norm = matplotlib.colors.Normalize(vmin=0, vmax=4), cmap='viridis_r'
              )},
          **kwargs):
    with Plot_RasterToTable(**kwargs) as ses:
        result= ses.plot_set(pars_d) 
        
        
    return result
if __name__=='__main__':
    #basic standalone setup
    logging.basicConfig(force=True, #overwrite root handlers
        stream=sys.stdout, #send to stdout (supports colors)
        level=logging.INFO, #lowest level to display
        )
 
    
    run01(logger = logging.getLogger())
    
 
    print('done')
