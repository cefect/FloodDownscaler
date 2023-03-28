'''
Created on Mar. 28, 2023

@author: cefect
'''


class Plot_samples_wrkr(PostBase):
    
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
    
 