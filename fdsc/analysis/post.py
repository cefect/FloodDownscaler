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
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

from hp.plot import Plotr, get_dict_str
from hp.pd import view



from fdsc.analysis.valid import ValidateSession

def dstr(d):
    return pprint.pformat(d, width=30, indent=0.3, compact=True, sort_dicts =False)

class Plot_rlays_wrkr(object):
    
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
                            fp_lib[k0][k2]=d2                    
                        
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
                    else:
                        pass
             
        #=======================================================================
        # get validation
        #=======================================================================
        fp_lib = {**{'vali':{'dep1':dep1V}}, **fp_lib} #order matters
 
 
        log.info('got fp_lib:\n%s\n\nmetric_lib:\n%s'%(dstr(fp_lib), dstr(metric_lib)))
        
 
        return fp_lib, metric_lib
        
    
    def plot_rlay_mat(self,
                      fp_lib, metric_lib=None, 
                      figsize=(9,5),
            **kwargs):
        """matrix plot comparing methods for downscaling: rasters
        
        rows: 
            valid, methods
        columns
            depthRaster r2, depthRaster r1, confusionRaster
        """
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('rlayMat',ext='.png', **kwargs)
        
        log.info(f'on {list(fp_lib.keys())}')
        
        cc_d = self.confusion_codes.copy()
        #=======================================================================
        # setup figure
        #=======================================================================
        row_keys = list(fp_lib.keys())
        col_keys = list(fp_lib[list(fp_lib.keys())[-1]].keys())
        
 
        fig, ax_d = self.get_matrix_fig(row_keys, col_keys, logger=log, 
                                        set_ax_title=False, figsize=figsize,
                                        constrained_layout=True,
                                        )
        
        
        #=======================================================================
        # colormap
        #=======================================================================
        cval_d = {
            cc_d['FN']:'#c700fe', cc_d['FP']:'#ff5101', cc_d['TP']:'#00fe19'
            }
        
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
        # plot loop
        #=======================================================================
        axImg_d = dict() #container for objects for colorbar
        for rowk, d0 in ax_d.items():
            for colk, ax in d0.items():                
                if not colk in fp_lib[rowk]:
                    continue 
                
                fp=fp_lib[rowk][colk]
                log.info(f'plotting {rowk}x{colk}: {os.path.basename(fp)}')
                with rio.open(fp, mode='r') as ds:
                    ar_raw = ds.read(1, window=None, masked=True)
                    
                    #===========================================================
                    # #apply masks
                    #===========================================================
                    if 'dep' in colk:
                        ar = np.where(ar_raw==0, np.nan, ar_raw)
                    elif 'confuGrid' in colk:
                        #mask out true negatives
                        ar = np.where(ar_raw==cc_d['TN'], np.nan, ar_raw)
                    else:
                        ar = ar_raw.data
                        
                    #===========================================================
                    # #get styles by key
                    #===========================================================
                    if 'confuGrid_fp' ==colk:
                        cmap=confuGrid_cmap
                        norm=confuGrid_norm
                    else:
                        cmap='viridis'
                        norm=None
                     
                    #===========================================================
                    # #raster plot
                    #===========================================================
                    #===========================================================
                    # _ = show(ar, 
                    #               transform=ds.transform, 
                    #               ax=ax,contour=False, cmap=cmap, interpolation='nearest', norm=norm)
                    #===========================================================
                    
                    ax_img=ax.imshow(ar, cmap=cmap, interpolation='nearest', norm=norm)
                    
                    """
                    help(show)
                    """
                    #===========================================================
                    # post format
                    #===========================================================
                    #hide labels
                    ax.get_xaxis().set_ticks([])
                    ax.get_yaxis().set_ticks([])
                    
                    #add text
                    if colk=='confuGrid_fp' and isinstance(metric_lib, dict):
                        ax.text(0.1, 0.9, get_dict_str(metric_lib[rowk]), 
                                transform=ax.transAxes, va='top', fontsize=6, color='black')
                        
                    #colorbar
                    if rowk==row_keys[-1]:
                        axImg_d[colk]=ax_img  
                         
                        
                        
        #help(fig.colorbar)
        #=======================================================================
        # post format
        #=======================================================================
 
        for rowk, d0 in ax_d.items():
            for colk, ax in d0.items():
                #turn off useless axis
                if rowk =='vali':
                    if not colk=='dep1':
                        #ad dummy plot
                        #ax.plot([0,1],[1,0])
                        #ax.imshow(ar)
                        #ax.cla()
                        #ax.set_axis_off()

                        #continue
                        
                        #colorbar
                        if 'dep' in colk:
                            spacing='proportional'
                            label='WSH (m)'
                            fmt = matplotlib.ticker.FuncFormatter(lambda x, p:'%.1f' % x)
                        else:
                            """not sure colorbar is best option here"""
                            #spacing='proportional'
                            spacing='uniform'
                            label=''
                            fmt=None
                            #fmt = matplotlib.ticker.FuncFormatter(lambda x, p:cc_di[x])
                        
                        
                        # Add an Axes to the right of the main Axes.
                        #ax1_divider = make_axes_locatable(ax)
                        #cax1 = ax1_divider.append_axes("bottom", size="50%", pad="10%")
                        cax1=ax
 
                        
                        cbar = fig.colorbar(axImg_d[colk],
                                            cax=cax1, 
                                            #ax=ax,
                                            orientation='horizontal',
                                     #ax=ax, location='bottom', # steal space from here
                                     extend='both', #pointed ends
                                     format=fmt, label=label,spacing=spacing,
                                     #shrink=0.8,
                                     )
                        

                        

                        
                
                #first col
                if colk==col_keys[0]:
                    ax.set_ylabel(rowk)
                    
                #last row
                if rowk==row_keys[0]:
                    ax.set_title({
                        'dep1':'WSH (r02)',
                        'dep2':'WSH (r32)',
                        'confuGrid_fp':'confusion'
                        }[colk])
 
                    
        #=======================================================================
        # wrap
        #=======================================================================
        log.info('finished')
        return self.output_fig(fig, ofp=ofp, logger=log)
        """
        plt.show()
        """
                    
                
                
                
                
        
        
        
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
        row_keys = ['true', 'nodp', 'cgs']#list(df.columns)
        col_keys = ['raw_hist', 'diff_hist', 'corr_scatter']
        
        
        fig, ax_d = self.get_matrix_fig(row_keys, col_keys, logger=log, 
                                        set_ax_title=False, figsize=figsize,
                                        constrained_layout=True,
                                        sharex=True, add_subfigLabel=True,
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
                ax.text(0.1, 0.9, get_dict_str(txt_d), 
                                transform=ax.transAxes, va='top', fontsize=8, color='black')
                 
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
                    ax.set_ylabel('pred depth (m)')
                    
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
    sim_color_d = {'true':'black', 'nodp':'orange', 'cgs':'teal'}
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
            assert os.path.exists(fp)
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
        
    

            
        
        
        
        
 
