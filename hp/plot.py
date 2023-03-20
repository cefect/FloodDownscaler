'''
Created on Feb. 15, 2021

@author: cefect

2022-08-03
simplified this a lot
now we should setup matplotlib (and defaults) in the caller script
'''


#==========================================================================
# logger setup-----------------------
#==========================================================================
import datetime



#==============================================================================
# imports------------
#==============================================================================
import os, string
import numpy as np
#import pandas as pd
#import scipy.stats

#==============================================================================
# # custom
#==============================================================================
 
from hp.basic import get_dict_str
 

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex
cm = 1 / 2.54
 


class Plotr(object):
 
    #===========================================================================
    # controls
    #===========================================================================
    fignum = 0 #counter for figure numbers
    

    
  
    
    def __init__(self, 
                 output_format='svg', 
                 add_stamp=True, 
                 transparent=True,
                 add_subfigLabel=True,                 
                 **kwargs):
        
        self.output_format=output_format
        self.add_stamp=add_stamp
        self.transparent=transparent
        self.add_subfigLabel=add_subfigLabel
        
        super().__init__(**kwargs)
        
 
    #===========================================================================
    # plotters------
    #===========================================================================
    
    def ax_data(self,  #add a plot of some data to an axis using kwargs
            ax, data_d,
            plot_type='hist', 
            
            #histwargs
            bins=20, 
            bin_lims=None,
            rwidth=0.9, 
            mean_line=None, #plot a vertical line on the mean
            hrange=None, #xlimit the data
            density=False,
            
            #boxkwargs
            add_box_cnts=True, #add n's
            color=None,
            
            #violin kwargs
            violin_line_kwargs=dict(color='black', alpha=0.5, linewidth=0.75),
            
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
            assert density, 'for plot_type==gaussian_kde'
        
        #check keys
        if not color_d is None:
            miss_l = set(color_d.keys()).symmetric_difference(data_d.keys())
            if not len(miss_l)==0:
                log.warning('color data key-mismatch: %s'%miss_l)
                color_d = {k:v for k,v in color_d.items() if k in data_d}
                
        if not color is None:
            assert color_d is None
        
        #===================================================================
        # HIST------
        #===================================================================
 
        if plot_type == 'hist':
            #setup bins
            if not bin_lims is None:
                binsi = np.linspace(bin_lims[0], bin_lims[1], bins)
            else:
                binsi = bins
            
            
            bval_ar, bins_ar, patches = ax.hist(
                data_d.values(), 
                range=hrange,
                bins=binsi, 
                density=density, # integral of the histogram will sum to 1
                color=color_d.values(), 
                rwidth=rwidth, 
                label=list(data_d.keys()), 
                **kwargs)
 
            #vertical mean line
            if not mean_line is None:
                ax.axvline(mean_line, color='black', linestyle='dashed')
 
            meta_d.update({'bin_max':bval_ar.max(), 
                           #'bin_cnt':bval_ar.shape[1] #ar.shape[0] =  number of groups
                           })

        #===================================================================
        # box plots--------
        #===================================================================
        elif plot_type == 'box':

            #===============================================================
            # #add bars
            #===============================================================
            boxres_d = ax.boxplot(data_d.values(), labels=data_d.keys(), meanline=True,
                    boxprops={'color':color},
                    whiskerprops={'color':color},
                    flierprops={'markeredgecolor':color, 'markersize':3,'alpha':0.5},
                **kwargs)
            #===============================================================
            # add extra text
            #===============================================================
            # counts on median bar
            if add_box_cnts:
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
                showmeans=True, widths=0.9,
                showextrema=False, **kwargs)
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
                pc.set_alpha(0.9)
            
            #style lines
            for partName in ['cmeans', 'cbars', 'cmins', 'cmaxes']:
                if partName in parts_d:
                    parts_d[partName].set(**violin_line_kwargs)
                
        #=======================================================================
        # gauisian KDE-----
        #=======================================================================
        elif plot_type=='gaussian_kde':
            first=True
            for dname, data in data_d.items():
                if label_key is None: label=None
                else:
                    label = '%s=%s'%(label_key, dname)
                log.info('    gaussian_kde on %i'%len(data))
                #filter
                if not hrange is None:
                    ar = data[np.logical_and(data>=hrange[0], data<=hrange[1])]
                else:
                    ar = data
                
   
                assert len(ar)>10
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
                    
                #stats
                if first:
                    hmin=min(ar)
                    hmax=max(ar)
                else:
                    hmin=min(ar.min(), hmin)
                    hmax = max(ar.max(), hmax)
            #post
            meta_d.update({'hmin':hmin, 'hmax':hmax})
        
        else:
            raise KeyError(plot_type)
        
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
        
 
    
    def get_matrix_fig(self, #conveneince for getting 
                       row_keys, #row labels for axis
                       col_keys, #column labels for axis (1 per column)
                       
                       fig_id=0,
                       figsize=None, #None: calc using figsize_scaler if present
                       figsize_scaler=None,
                        #tight_layout=False,
                        constrained_layout=True,
                        set_ax_title=True, #add simple axis titles to each subplot
                        logger=None,
                        add_subfigLabel=None,
                        fig=None,
                        **kwargs):
        
        """get a matrix plot with consistent object access
        
        Parameters
        ---------
        figsize_scaler: int
            multipler for computing figsize from the number of col and row keys
            
        add_subfigLabel: bool
            add label to each axis (e.g., A1)
            
        Returns
        --------
        dict
            {row_key:{col_key:ax}}
            
        """
        
        
        #=======================================================================
        # defautls
        #=======================================================================
        if logger is None: logger=self.logger
        log=logger.getChild('get_mat_fig')
        if add_subfigLabel is None: add_subfigLabel=self.add_subfigLabel
        #special no singluar columns
        if col_keys is None: ncols=1
        else:ncols=len(col_keys)
        

        
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
        if fig is None:
            if figsize is None: 
                if figsize_scaler is None:
                    figsize=matplotlib.rcParams['figure.figsize']
                else:
                    
                    figsize = (len(col_keys)*figsize_scaler, len(row_keys)*figsize_scaler)
                    
                    #fancy diagnostic p rint
                    fsize_cm = tuple(('%.2f cm'%(e/cm) for e in figsize))                    
                    log.info(f'got figsize={fsize_cm} from figsize_scaler={figsize_scaler:.2f} and col_cnt={len(col_keys)}')
                    
 
                
        
            fig = plt.figure(fig_id,
                figsize=figsize,
                #tight_layout=tight_layout,
                constrained_layout=constrained_layout,
 
                )
        else:
            #check the user doesnt expect to create a new figure
            assert figsize_scaler is None
            assert figsize is None
            assert constrained_layout is None
            assert fig_id is None
        

        #=======================================================================
        # add subplots
        #=======================================================================
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
        
                #=======================================================================
                # post format
                #=======================================================================
                if set_ax_title:
                    if col_keys[j] == '':
                        ax_title = row_keys[i]
                    else:
                        ax_title='%s.%s'%(row_keys[i], col_keys[j])
                    
                    ax.set_title(ax_title)
                    
                    
                if add_subfigLabel:
                    letter=list(string.ascii_lowercase)[j]
                    ax.text(0.05, 0.95, 
                            '(%s%s)'%(letter, i), 
                            transform=ax.transAxes, va='top', ha='left',
                            size=matplotlib.rcParams['axes.titlesize'],
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", lw=0.0,alpha=0.5 ),
                            )
 
                
            
 
        log.info('built %ix%i w/ figsize=%s'%(len(col_keys), len(row_keys), figsize))
        return fig, ax_d
            
 
    def _build_color_d(self, keys,
                       cmap = plt.cm.get_cmap(name='Set1')
                       ):
        """get a dict of key:hex-color from the list of keys and the colormap"""
        
        ik_d = dict(zip(keys, np.linspace(0, 1, len(keys))))
 
        color_d = {k:rgb2hex(cmap(ni)) for k, ni in ik_d.items()}
        return color_d
    
    #===========================================================================
    # OUTPUTTRS------
    #===========================================================================
    def output_fig(self, 
                   fig,
                   
                   #file controls
                   out_dir = None,  
                   ofp=None, #defaults to figure name w/ a date stamp
                   fname = None, #filename
                   clean_ofp=True,
                   
                   #stylig
                   add_stamp=None, #whether to write the filepath as a small stamp
                   
                   #figure write controls
                 fmt=None, 
                  transparent=None, 
                  dpi = 300,
                  logger=None,
                  **kwargs):
        #======================================================================
        # defaults
        #======================================================================
        
        #if overwrite is None: overwrite = self.overwrite
        if logger is None: logger=self.logger
        #if fmt is None: fmt = self.output_format
        if add_stamp is None: add_stamp=self.add_stamp
        log = logger.getChild('output_fig')
        
        if add_stamp is None: add_stamp=self.add_stamp
        if transparent is None: transparent=self.transparent
        
        #=======================================================================
        # precheck
        #=======================================================================
        
        assert isinstance(fig, matplotlib.figure.Figure)
        log.debug('on %s'%fig)
        #======================================================================
        # filepath
        #======================================================================
        if ofp is None:
            if fmt is None: fmt=self.output_format
            if out_dir is None: out_dir = self.out_dir
            if not os.path.exists(out_dir):os.makedirs(out_dir)
            
            #file setup
            if fname is None:
                try:
                    fname = fig._suptitle.get_text()
                except:
                    fname = self.run_name
                    
                fname =str('%s_%s'%(fname, self.fancy_name)).replace(' ','')
                
            ofp = os.path.join(out_dir, '%s.%s'%(fname, fmt))
        else:
 
            assert fmt is None, 'can not specify \'fmt\' and \'ofp\''
 
            fmt = os.path.splitext(ofp)[1].replace('.', '')
            assert not fmt=='', 'forget the period?'
            
            
        if clean_ofp:
            for s in [',', ';', ')', '(', '=', ' ', '\'']:
                ofp = ofp.replace(s,'')
 
        #=======================================================================
        # plot stamp
        #=======================================================================
        if add_stamp:
 
            txt = '%s (%s)'%(os.path.basename(ofp), datetime.datetime.now().strftime('%Y-%m-%d'))
            fig.text(1,0, txt, fontsize=2, color='black', alpha=0.5, ha='right', va='bottom')
        #=======================================================================
        # #write the file
        #=======================================================================
        try: 
            fig.savefig(ofp, dpi = dpi, format = fmt, transparent=transparent, **kwargs)
            log.info('saved figure to file:\n   %s'%ofp)
        except Exception as e:
            raise IOError('failed to write figure to file w/ \n    %s'%e)
        
        plt.close()
        return ofp
    
def hide_text(ax):
    """hide all text objects foundon the axis"""
    for obj in ax.get_children():
        if isinstance(obj, matplotlib.text.Text):
            obj.set_visible(False)
    
    