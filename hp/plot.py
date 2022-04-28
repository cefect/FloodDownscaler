'''
Created on Feb. 15, 2021

@author: cefect
'''


#==========================================================================
# logger setup-----------------------
#==========================================================================
import logging, configparser, datetime



#==============================================================================
# imports------------
#==============================================================================
import os
import numpy as np
import pandas as pd
import scipy.stats

#==============================================================================
# # custom
#==============================================================================
from hp.exceptions import Error

from hp.pd import view
from hp.oop import Basic

import matplotlib


class Plotr(Basic):
    
    #===========================================================================
    # parameters from control file
    #===========================================================================
    #[plotting]

    color = 'black'
    linestyle = 'dashdot'
    linewidth = 2.0
    alpha =     0.75        #0=transparent 1=opaque
    marker =    'o'
    markersize = 4.0
    fillstyle = 'none'    #marker fill style
    impactfmt_str = '.2e'
        #',.0f' #Thousands separator
        
    impactFmtFunc = None
    
    #===========================================================================
    # controls
    #===========================================================================
    fignum = 0 #counter for figure numbers
    
    #===========================================================================
    # defaults
    #===========================================================================
    val_str='*default'
        
    """values are dummies.. upd_impStyle will reset form attributes"""
    impStyle_d = {
            'color': 'black',
            'linestyle': 'dashdot',
            'linewidth': 2.0,
            'alpha':0.75 , # 0=transparent, 1=opaque
            'marker':'o',
            'markersize':  4.0,
            'fillstyle': 'none' #marker fill style
                            }

    
    def __init__(self,

 
                 
                 #init controls
                 init_plt_d = {}, #container of initilzied objects
 
                  #format controls
                  grid = True, logx = False, 
                  
                  
                  #figure parametrs
                figsize     = (6.5, 4), 
                transparent = False,
                    
                #hatch pars
                    hatch =  None,
                    h_color = 'blue',
                    h_alpha = 0.1,
                    
                    #impactFmtFunc=None, #function for formatting the impact results
                        
                        #Option1: pass a raw function here
                        #Option2: pass function to init_fmtFunc
                        #Option3: use 'impactfmt_str' kwarg to have init_fmtFunc build
                            #default for 'Model' classes (see init_model)


                 **kwargs
                 ):
        


        
        super().__init__( **kwargs) #initilzie teh baseclass

        #=======================================================================
        # attached passed        
        #=======================================================================

        self.plotTag = self.tag #easier to store in methods this way
 
        self.grid    =grid
        self.logx    =logx
 
        self.figsize    =figsize
        self.hatch    =hatch
        self.h_color    =h_color
        self.h_alpha    =h_alpha
        self.transparent=transparent
        
        #init matplotlib
        """TODO: need a simpler way to handle this"""
        if init_plt_d is None:
            pass
        elif len(init_plt_d)==0:
            self.init_plt_d = self._init_plt() #setup matplotlib
        else:
            for k,v in init_plt_d.items():
                setattr(self, k, v)
                
            self.init_plt_d = init_plt_d
        

            

        
        
        self.logger.debug('init finished')
        
        """call explicitly... sometimes we want lots of children who shouldnt call this
        self._init_plt()"""
        

    
    def _init_plt(self,  #initilize matplotlib
                #**kwargs
                  ):
        """
        calling this here so we get clean parameters each time the class is instanced
        
        
        """

        
        #=======================================================================
        # imports
        #=======================================================================
        import matplotlib
        matplotlib.use('Qt5Agg') #sets the backend (case sensitive)
        matplotlib.set_loglevel("info") #reduce logging level
        import matplotlib.pyplot as plt
        
        #set teh styles
        plt.style.use('default')
        
        #font
        matplotlib_font = {
                'family' : 'serif',
                'weight' : 'normal',
                'size'   : 8}
        
        matplotlib.rc('font', **matplotlib_font)
        matplotlib.rcParams['axes.titlesize'] = 10 #set the figure title size
        
        #spacing parameters
        matplotlib.rcParams['figure.autolayout'] = False #use tight layout
        
        #legends
        matplotlib.rcParams['legend.title_fontsize'] = 'large'
        
        self.plt, self.matplotlib = plt, matplotlib
        
        self.logger.info('matplotlib version = %s'%matplotlib.__version__)
        
        #=======================================================================
        # seaborn
        #=======================================================================
        import seaborn as sns
        self.sns = sns
        
        return {'plt':plt, 'matplotlib':matplotlib, 'sns':sns}
    

        
    #===========================================================================
    # plotters------
    #===========================================================================
    
    def ax_data(self,  #add a plot of some data to an axis using kwargs
            ax, data_d,
            plot_type='hist', 
            
            #histwargs
            bins=20, rwidth=0.9, 
            mean_line=None, #plot a vertical line on the mean
            hrange=None, #xlimit the data
            density=False,
            
            #styling
            zero_line=False,
            color_d = None,
            label_key=None,
            
            logger=None, **kwargs):
                
        """as we use these methods in a few funcs... decided to factor
        
        plt.show()
        """
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log=logger.getChild('ax_data')
        meta_d=dict()
        
        if plot_type=='gaussian_kde':
            assert density
        
        #check keys
        miss_l = set(color_d.keys()).symmetric_difference(data_d.keys())
        assert len(miss_l)==0, 'color data key-mismatch: %s'%miss_l
        
        #===================================================================
        # HIST------
        #===================================================================
 
        if plot_type == 'hist':
            bval_ar, bins_ar, patches = ax.hist(
                data_d.values(), 
                range=hrange,
                bins=bins, 
                density=density, # integral of the histogram will sum to 1
                color=color_d.values(), 
                rwidth=rwidth, 
                label=list(data_d.keys()), 
                **kwargs)
 
            #vertical mean line
            if not mean_line is None:
                ax.axvline(mean_line, color='black', linestyle='dashed')
            
            #ar.size
            
            meta_d.update({'bin_max':bval_ar.max(), 
                           #'bin_cnt':bval_ar.shape[1] #ar.shape[0] =  number of groups
                           
                           })
 
            
        #===================================================================
        # box plots--------
        #===================================================================
        elif plot_type == 'box':
            #===============================================================
            # zero line
            #===============================================================
            #ax.axhline(0, color='black')
            #===============================================================
            # #add bars
            #===============================================================
            boxres_d = ax.boxplot(data_d.values(), labels=data_d.keys(), meanline=True,**kwargs)
            # boxprops={'color':newColor_d[rowVal]},
            # whiskerprops={'color':newColor_d[rowVal]},
            # flierprops={'markeredgecolor':newColor_d[rowVal], 'markersize':3,'alpha':0.5},
            #===============================================================
            # add extra text
            #===============================================================
            # counts on median bar
            for gval, line in dict(zip(data_d.keys(), boxres_d['medians'])).items():
                x_ar, y_ar = line.get_data()
                ax.text(x_ar.mean(), y_ar.mean(), 'n%i' % len(data_d[gval]), 
                # transform=ax.transAxes,
                    va='bottom', ha='center', fontsize=8)
        
        #===================================================================
        # violin plot-----
        #===================================================================
        elif plot_type == 'violin':
            #===============================================================
            # plot
            #===============================================================
            parts_d = ax.violinplot(data_d.values(), 
                showmeans=True, 
                showextrema=True, **kwargs)
            #===============================================================
            # color
            #===============================================================
            #===============================================================
            # if len(data_d)>1:
            #     """nasty workaround for labelling"""
            #===============================================================
 
            #ckey_d = {i:color_key for i,color_key in enumerate(labels)}
            #style fills
            
            for dname, pc in dict(zip(data_d.keys(), parts_d['bodies'])).items():
 
                pc.set_facecolor(color_d[dname])
                pc.set_edgecolor(color_d[dname])
                pc.set_alpha(0.5)
            
            #style lines
            for partName in ['cmeans', 'cbars', 'cmins', 'cmaxes']:
                parts_d[partName].set(color='black', alpha=0.5)
                
        #=======================================================================
        # gauisian KDE-----
        #=======================================================================
        elif plot_type=='gaussian_kde':
            for dname, data in data_d.items():
                if label_key is None: label=None
                else:
                    label = '%s=%s'%(label_key, dname)
                log.info('    gaussian_kde on %i'%len(data))
                #filter
                #===============================================================
                # if not hrange is None:
                #     
                #     ar = data[np.logical_and(data>hrange[0], data<=hrange[1])]
                # else:
                #     ar = data
                #===============================================================
                
                ar = data  
                
                #build teh function
                kde = scipy.stats.gaussian_kde(ar, 
                                                   bw_method='scott',
                                                   weights=None, #equally weighted
                                                   )
                
                #plot it
                xvals = np.linspace(ar.min()+.01, ar.max(), 500)
                ax.plot(xvals, kde(xvals), color=color_d[dname], label=label, **kwargs)
                
                #vertical mean line
                if not mean_line is None:
                    ax.axvline(mean_line, color='black', linestyle='dashed')
        
        else:
            raise Error(plot_type)
        
        #=======================================================================
        # post----
        #=======================================================================
        if zero_line:
            ax.axhline(0.0, color='black', linestyle='solid', linewidth=0.5)
        
        
 
        
        return  meta_d


        
        

    def _postFmt(self, #text, grid, leend
                 ax, 

                 
                 grid=None,
                 
                 #plot text
                 val_str=None,
                 xLocScale=0.1, yLocScale=0.1,
                 
                 #legend kwargs
                 legendLoc = 1,
                 
                 legendHandles=None, 
                 legendTitle=None,
                 ):
        
        #=======================================================================
        # defaults
        #=======================================================================
        plt, matplotlib = self.plt, self.matplotlib
        if grid is None: grid=self.grid
        
        #=======================================================================
        # Add text string 'annot' to lower left of plot
        #=======================================================================
        if isinstance(val_str, str):
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            
            x_text = xmin + (xmax - xmin)*xLocScale # 1/10 to the right of the left axis
            y_text = ymin + (ymax - ymin)*yLocScale #1/10 above the bottom axis
            anno_obj = ax.text(x_text, y_text, val_str)
        
        #=======================================================================
        # grid
        #=======================================================================
        if grid: ax.grid()
        

        #=======================================================================
        # #legend
        #=======================================================================
        if isinstance(legendLoc, int):
            if legendHandles is None:
                h1, l1 = ax.get_legend_handles_labels() #pull legend handles from axis 1
            else:
                assert isinstance(legendHandles, tuple)
                assert len(legendHandles)==2
                h1, l1 = legendHandles
            #h2, l2 = ax2.get_legend_handles_labels()
            #ax.legend(h1+h2, l1+l2, loc=2) #turn legend on with combined handles
            ax.legend(h1, l1, loc=legendLoc, title=legendTitle) #turn legend on with combined handles
        
        return ax
    
    def _tickSet(self,
                 ax,
                 xfmtFunc=None, #function that returns a formatted string for x labels
                 xlrot=0,
                 
                 yfmtFunc=None,
                 ylrot=0):
        

        #=======================================================================
        # xaxis
        #=======================================================================
        if not xfmtFunc is None:
            # build the new ticks
            l = [xfmtFunc(value) for value in ax.get_xticks()]
                  
            #apply the new labels
            ax.set_xticklabels(l, rotation=xlrot)
        
        
        #=======================================================================
        # yaxis
        #=======================================================================
        if not yfmtFunc is None:
            # build the new ticks
            l = [yfmtFunc(value) for value in ax.get_yticks()]
                  
            #apply the new labels
            ax.set_yticklabels(l, rotation=ylrot)
        
    def _get_val_str(self, #helper to get value string for writing text on the plot
                     val_str, #cant be a kwarg.. allowing None
                     impactFmtFunc=None,
                     ):
        """
        generally just returns the val_str
            but also provides some special handles
        """
        #=======================================================================
        # defaults
        #=======================================================================
        if impactFmtFunc is None: impactFmtFunc=self.impactFmtFunc
        if val_str is None:
            val_str = self.val_str
        
        #=======================================================================
        # special keys
        #=======================================================================
        if isinstance(val_str, str):
            if val_str=='*default':
                assert isinstance(self.ead_tot, float)
                val_str='total annualized impacts = ' + impactFmtFunc(self.ead_tot)
            elif val_str=='*no':
                val_str=None
            elif val_str.startswith('*'):
                raise Error('unrecognized val_str: %s'%val_str)
                
        return val_str
    
    def get_matrix_fig(self, #conveneince for getting a matrix plot with consistent object access
                       row_keys, #row labels for axis
                       col_keys, #column labels for axis (1 per column)
                       
                       fig_id=0,
                       figsize=None, #None: calc using figsize_scaler if present
                       figsize_scaler=None,
                        tight_layout=False,
                        constrained_layout=True,
                        set_ax_title=False, #add simple axis titles to each subplot
                        **kwargs):
        
        
        #=======================================================================
        # defautls
        #=======================================================================
        #special no singluar columns
        if col_keys is None: ncols=1
        else:ncols=len(col_keys)
        
        if figsize is None: 
            if figsize_scaler is None:
                figsize=self.figsize
            else:
                figsize = (len(col_keys)*figsize_scaler, len(row_keys)*figsize_scaler)
        
        #=======================================================================
        # precheck
        #=======================================================================
        """needs to be lists (not dict keys)"""
        assert isinstance(row_keys, list)
        #assert isinstance(col_keys, list)
        #=======================================================================
        # build figure
        #=======================================================================
        # populate with subplots
        fig = self.plt.figure(fig_id,
            figsize=figsize,
            tight_layout=tight_layout,
            constrained_layout=constrained_layout,
            )
        

        
        ax_ar = fig.subplots(nrows=len(row_keys), ncols=ncols, **kwargs)
        
        #convert to array
        if not isinstance(ax_ar, np.ndarray):
            assert len(row_keys)==len(col_keys)
            assert len(row_keys)==1
            
            ax_ar = np.array([ax_ar])
            
        
        #=======================================================================
        # convert to dictionary
        #=======================================================================
        ax_d = dict()
        for i, row_ar in enumerate(ax_ar.reshape(len(row_keys), len(col_keys))):
            ax_d[row_keys[i]]=dict()
            for j, ax in enumerate(row_ar.T):
                ax_d[row_keys[i]][col_keys[j]]=ax
                
                if set_ax_title:
                    if col_keys[j] == '':
                        ax_title = row_keys[i]
                    else:
                        ax_title='%s.%s'%(row_keys[i], col_keys[j])
                    
                    ax.set_title(ax_title)
                
            
 
            
        return fig, ax_d
            
    def get_color_d(self,
                    cvals,
                    colorMap=None,
                    ):
                    
        if colorMap is None: colorMap=self.colorMap
        cmap = self.plt.cm.get_cmap(name=colorMap) 
        return {k:matplotlib.colors.rgb2hex(cmap(ni)) for k, ni in dict(zip(cvals, np.linspace(0, 1, len(cvals)))).items()}
    
    #===========================================================================
    # OUTPUTTRS------
    #===========================================================================
    def output_fig(self, 
                   fig,
                   
                   #file controls
                   out_dir = None, overwrite=None, 
                   out_fp=None, #defaults to figure name w/ a date stamp
                   fname = None, #filename
                   
                   #stylig
                   add_stamp=True, #whether to write the filepath as a small stamp
                   
                   #figure write controls
                 fmt='svg', 
                  transparent=None, 
                  dpi = 300,
                  logger=None,
                  ):
        #======================================================================
        # defaults
        #======================================================================
        if out_dir is None: out_dir = self.out_dir
        if overwrite is None: overwrite = self.overwrite
        if logger is None: logger=self.logger
        log = logger.getChild('output_fig')
        
        if transparent is None: transparent=self.transparent
        
        if not os.path.exists(out_dir):os.makedirs(out_dir)
        #=======================================================================
        # precheck
        #=======================================================================
        
        assert isinstance(fig, matplotlib.figure.Figure)
        log.debug('on %s'%fig)
        #======================================================================
        # filepath
        #======================================================================
        if out_fp is None:
            #file setup
            if fname is None:
                try:
                    fname = fig._suptitle.get_text()
                except:
                    fname = self.name
                    
                fname =str('%s_%s'%(fname, self.resname)).replace(' ','')
                
            out_fp = os.path.join(out_dir, '%s.%s'%(fname, fmt))
            
        if os.path.exists(out_fp): 
            assert overwrite
            os.remove(out_fp)
            
            

        """
        fig.show()
        """
        #=======================================================================
        # plot stamp
        #=======================================================================
        if add_stamp:
 
            txt = '%s (%s)'%(os.path.basename(out_fp), datetime.datetime.now().strftime('%Y-%m-%d'))
            fig.text(1,0, txt, fontsize=6, color='black', alpha=0.5, ha='right', va='bottom')
        #=======================================================================
        # #write the file
        #=======================================================================
        try: 
            fig.savefig(out_fp, dpi = dpi, format = fmt, transparent=transparent)
            log.info('saved figure to file:\n   %s'%out_fp)
        except Exception as e:
            raise Error('failed to write figure to file w/ \n    %s'%e)
        
        return out_fp
    