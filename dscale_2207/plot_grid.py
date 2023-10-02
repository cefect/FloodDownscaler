'''
Created on Jan. 9, 2023

@author: cefect

plot a raster as a table/grid
    manuscripts toy grids
'''
import os, logging, sys, pprint
 
import numpy as np
import rasterio as rio
from rasterio.plot import show
 

from hp.plot import Plotr
from hp.oop import Session

#===============================================================================
# setup matplotlib----------
#===============================================================================
cm = 1/2.54

output_format='svg'
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
font_size=12
matplotlib.rc('font', **{'family' : 'sans-serif','sans-serif':'Tahoma','weight' : 'normal','size':font_size})
 
 
for k,v in {
    'axes.titlesize':10,
    'axes.labelsize':10,
    'xtick.labelsize':8,
    'ytick.labelsize':8,
    'figure.titlesize':12,
    'figure.autolayout':False,
    'figure.figsize':(8*cm,8*cm),
    'legend.title_fontsize':'large',
    'text.usetex':usetex,
    }.items():
        matplotlib.rcParams[k] = v
  
print('loaded matplotlib %s' % matplotlib.__version__)
    


class Plot_RasterToTable(Plotr, Session):
    def __init__(self,
                 proj_name='RasterToTable',
                 **kwargs):
        
        super().__init__(proj_name=proj_name, **kwargs)
         
    def plot_set(self, pars_lib, **kwargs):
        """plot a set of rasters"""
        #=======================================================================
        # defaults
        #=======================================================================
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('Plot_RasterToTable',ext='.png', **kwargs)
        
        res_d=dict()
        
        #=======================================================================
        # loop and plot
        #=======================================================================
        log.info(f'plotting {len(pars_lib)} rasters')
        for name, pars_d in pars_lib.items():
            fp = pars_d.pop('fp')
            log.info(f'plotting {name} w/\n    {pars_d}')
            res_d[name] = self.plot_rasterToTable(fp, resname = self._get_resname(dkey=f'rasterTable_{name}'),
                                             logger=log.getChild(name), **pars_d)
            
        return res_d
        
    
    def plot_rasterToTable(self, fp, 
                           cmap='viridis_r', norm=None,
                           fontsize=font_size,
                           **kwargs):
        """main caller to plot the raster file"""
        assert os.path.exists(fp), fp
        
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('Plot_RasterToTable',ext='.'+output_format, **kwargs)
        
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
        
        table.set_fontsize(fontsize)
        
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
            
 
            
wse_plot_kwargs= dict( norm = matplotlib.colors.Normalize(vmin=2, vmax=5), cmap='Blues') 
        
    
def run_toy_0205(
        pars_lib = None,
        run_name='r1',
          **kwargs):
    """
    build plots of toy grid data
    
    """
    #===========================================================================
    # setup downsample parameters
    #===========================================================================
    if pars_lib is None:
        """where do these come from? data seems to be from hp.tests.data.toy_rasters.py"""
        fp_d = {
            #'none_wse1':r'l:\10_IO\2207_dscale\outs\fdsc\toy\200230205\none\FloodDownscaler_test_0205_dsc.tif',
            #'nodp':r'l:\10_IO\2207_dscale\outs\fdsc\toy\200230205\wetPartialsOnly\FloodDownscaler_test_0205_dsc.tif',
            'cgs_wse1':r'l:\10_IO\2207_dscale\outs\fdsc\toy\200230205\costGrowSimple\FloodDownscaler_test_0205_dsc.tif',
            #'cgs_costDistance':r'l:\10_IO\2207_dscale\outs\fdsc\toy\200230205\costGrowSimple\p2DP\costAllocation.tif',
            #'cgs_demFilter':r'l:\10_IO\2207_dscale\outs\fdsc\toy\200230205\costGrowSimple\p2DP\FloodDownscaler_test_0205_filter.tif',
            #'cgs_clump':r'l:\10_IO\2207_dscale\outs\fdsc\toy\200230205\costGrowSimple\p2DP\clump.tif',
            #'s14_wse1_rsmp':r'C:\Users\cefect\AppData\Local\Temp\pytest-of-cefect\pytest-4226\test_runr_schumann14_kwargs0_C0\temp_Session_0840\dsc\FloodDownscaler_test_0205_wse1_resamp.tif',
            #'s14_srch_mask':r'l:\10_IO\2207_dscale\outs\fdsc\toy\200230205\schumann14\FloodDownscaler_test_0205_srch.tif',
            #'s14_knn':r'l:\10_IO\2207_dscale\outs\fdsc\toy\200230205\schumann14\FloodDownscaler_test_0205_knnFill.tif',
            #'s14_wse1':r'l:\10_IO\2207_dscale\outs\fdsc\toy\200230205\schumann14\FloodDownscaler_test_0205_schu14.tif',
            }
 
        pars_lib=dict()
        for k,v in fp_d.items():
            pars_lib[k] = dict (**dict(fp=fp_d[k]), **wse_plot_kwargs)
            
            if '_mask' in k:
                pars_lib[k]['cmap'] = 'RdYlGn'
                pars_lib[k]['norm'] = matplotlib.colors.Normalize(vmin=0, vmax=1)
            
            
            elif k=='cgs_clump':
                pars_lib['cgs_clump']['cmap']='Set1'
                pars_lib['cgs_clump']['norm'] = matplotlib.colors.Normalize(vmin=0, vmax=3)
 
    #===========================================================================
    # build toy inputs
    #===========================================================================
    from tests.test_dsc import toy_d
    dem1_fp, wse2_fp = toy_d['dem1'], toy_d['wse2']
 
         
    # add these
    pars_lib.update(
        {'dem1':{'fp':dem1_fp, 'cmap':'copper_r', 'norm':matplotlib.colors.Normalize(vmin=1.0, vmax=6.0)},
         'wse2':{**{'fp':wse2_fp}, **wse_plot_kwargs}}
        )
    
    #===========================================================================
    # run
    #===========================================================================
    with Plot_RasterToTable(proj_name='toy',run_name=run_name, **kwargs) as ses:
        result= ses.plot_set(pars_lib) 
        
    d = {k: v.replace('\\', '/') for k, v in result.items()}
    
    print('\n\n\n')
    print(pprint.pformat(d, width=30, indent=True, compact=True, sort_dicts =False))
        
        
    return result



if __name__=='__main__':
    #basic standalone setup
    logging.basicConfig(force=True, #overwrite root handlers
        stream=sys.stdout, #send to stdout (supports colors)
        level=logging.INFO, #lowest level to display
        )
 
    
    run_toy_0205(logger = logging.getLogger())
    
 
    print('done')
