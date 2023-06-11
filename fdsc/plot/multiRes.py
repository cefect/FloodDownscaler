'''
Created on Jun. 10, 2023

@author: cefect
'''
import itertools, os
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

from hp.pd import view, nested_dict_to_dx, dict_to_multiindex

from fdsc.plot.control import Fdsc_Plot_Session







class Plot_MultiRes_Stats(object):
    
    def plot_multiRes_stats_single(self, df,
                            
 
                            label_d = None,
                            color_d=None,
                            
                            base_scale=None,
                            subplots_kwargs=dict(figsize=None),
                            
                            ylab='volume (m3)',
                            **kwargs):
        
        """plot single axis dot+lines using serx
        
        color: mdex(0)
        xvals:mdex(1)
        yvals: serx.values
        
 
        """
        
        #=======================================================================
        # setup
        #=======================================================================
        log, tmp_dir, out_dir, ofp, resname = self._func_setup(
            f'stats', ext='.'+self.output_format, **kwargs)
            
        if label_d is None: label_d={v:k for k,v in self.nicknames_d.items()}
        if color_d is None:
            color_d = self._build_color_d(df.index.values)
        #=======================================================================
        # setup figure
        #=======================================================================
        
        fig, ax = plt.subplots(**subplots_kwargs)
        
        markers = itertools.cycle((',', '+', 'o', '*'))
        
        #=======================================================================
        # plot
        #=======================================================================
        log.info('plotting')
        
        for simk, row in df.iterrows():
            row.plot(ax=ax, label = label_d[simk], color=color_d[simk], marker = next(markers),
                     linewidth=1.0, alpha=0.8)
        
        """
        plt.show()
        """
        #=======================================================================
        # post
        #=======================================================================
        ax.legend()
        
        ax.set_xticks(df.columns.values)
        ax.set_xlabel('downscale')
        ax.set_ylabel(ylab)
        ax.grid()
        
        #second xaxis
        if not base_scale is None:
            def scale_to_res(x):
                return base_scale/x
 
            
            secax = ax.secondary_xaxis('top', functions=(scale_to_res, scale_to_res))
            secax.set_xlabel('resolution (m)')
            
            # Transform the bottom x-axis tick positions to the top x-axis scale
            top_xticks = scale_to_res(df.columns).values
            
            #ax.get_xticks()
            # Set the top x-axis ticks and labels
            
            #l = [float(item.get_text()) for item in ax.get_xticklabels()]
            
            secax.set_xticks(top_xticks)
            secax.set_xticklabels(top_xticks)
             
        
        #=======================================================================
        # wraop
        #=======================================================================
        
        return self.output_fig(fig, ofp=ofp, logger=log, dpi=600)
        
 
 
 
class Fdsc_MultiRes_Plot_Session(Plot_MultiRes_Stats, Fdsc_Plot_Session):
    
    def load_run_serx_multiRes_stats(self, run_lib, **kwargs):
        """load a series from the results dictionary"""
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('load_run_serx_multiRes', **kwargs)
        
        #print contents
        k0=list(run_lib.keys())[1]
        #=======================================================================
        # print(f'\nfor {k1}')
        # print(run_lib[k1].keys())
        # print(pprint.pformat(run_lib[k1], width=30, indent=0.3, compact=True, sort_dicts =False))
        #=======================================================================
        
        #print keys
        print(run_lib.keys())
        for k1, d1 in run_lib[k0].items():
            print(d1.keys())
            for k2, d2 in d1.items():
                print(d2.keys())
                break
            
            break
            
            
        #ins_d = run_lib.pop('inputs')
        #get name converter
        ins_d = run_lib['inputs']['inputs1']
        print(ins_d['meta'].keys())
        dscale_nm_d = ins_d['meta']['dscale_nm_d'].copy()
 
        #=======================================================================
        # pull out stats
        #=======================================================================
        stats_lib = dict()
        meta_lib = dict()
        for sim_name, d0 in run_lib.items():
            if sim_name=='inputs': continue
            stats_lib[sim_name] = dict()
            meta_lib[sim_name]=dict()
            for dscale, d1 in d0.items():
                #print(dstr(d1['meta']['dem_raw']))

                stats_lib[sim_name][dscale] = d1['grid_stats']
                
                #add meta
                try:
                    md = {
                        'resolution':d1['meta']['dem_raw']['res'][0],
                        }
                except:
                    raise IOError('?')
                meta_lib[sim_name][dscale]=md
            
        
        #=======================================================================
        # convert to serx
        #=======================================================================
        #print(dstr(meta_lib))
        serx = pd.DataFrame(dict_to_multiindex(stats_lib), index=['val']).iloc[0, :]
        
 
        #fix the names
        serx.index.set_names(['simName', 'dscale', 'dataType', 'varName'], inplace=True)
 

        #=======================================================================
        # convert meta
        #=======================================================================
 
        meta_serx  = pd.DataFrame(dict_to_multiindex({'meta':meta_lib}), index=['val']).iloc[0, :]
        
        meta_serx.index.set_names(['dataType', 'simName', 'dscale', 'varName'], inplace=True)
        meta_serx = meta_serx.reorder_levels(serx.index.names)
        
        serx = pd.concat([serx, meta_serx]).sort_index()
        
        #----------------------------------------------------------------- clean
        #re-map level values from a dictionary
 
        serx.rename(dscale_nm_d, level=1, inplace=True) #downscale values
        
        serx.rename(self.nicknames_d, level=0, inplace=True) #change to short sim names 
        
 
        
        
        log.info(f'finished w/ {serx.index.to_frame().shape}')
        
        return serx
 
        
