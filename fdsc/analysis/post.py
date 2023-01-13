'''
Created on Jan. 9, 2023

@author: cefect

data analysis on multiple downscale results
'''


import logging, os, copy, datetime, pickle, pprint
import numpy as np
import pandas as pd
import rasterio as rio

from fdsc.analysis.valid import ValidateSession

def dstr(d):
    return pprint.pformat(d, width=30, indent=0.3, compact=True, sort_dicts =False)

class Plot_rlays_wrkr(object):
    
    def collect_rlay_fps(self, run_lib, **kwargs):
        """collect the filepaths from the run_lib"""
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('load_metas', **kwargs)
        
        fp_lib={k:dict() for k in run_lib.keys()}
        metric_lib = {k:dict() for k in run_lib.keys()}
        
        for k0, d0 in run_lib.items(): #simulation name
            for k1, d1 in d0.items(): #cat0
                for k2, d2 in d1.items():
                    if k1=='smry':
                        if k2 in ['wse1_fp', 'wse2']:
                            fp_lib[k0][k2]=d2                    
                        
                    elif k1=='vali':
                        if k2=='inun':
                            for k3, v3 in d2.items():
                                if k3=='confuGrid_fp':
                                    fp_lib[k0][k3]=v3
                                else:
                                    metric_lib[k0][k3]=v3
                    else:
                        pass
             
        
        #=======================================================================
        # #pull from sumamary
        # smry_d = {k:v['smry'] for k,v in run_lib.items()}
        # d = {k0:{k1:v1  for k1, v1 in v0.items() } for k0,v0 in smry_d.items()}
        # 
        # #pull validation data
        # vali_d = {k:v['vali'] for k,v in run_lib.items()}
        # inun_d = {k:v['inun'] for k,v in vali_d.items()}
        # 
        # #add rasters
        # metric_d = dict()
        # for k0,v0 in inun_d.items():
        #     for k1, v1 in v0.items():
        #         if k1=='confuGrid_fp':
        #             d[k1]=v1
        #         else:
        #             metric_d[k0]=v1
        #=======================================================================
        log.info('got fp_lib::\n%s\n\nmetric_lib:\n%s'%(dstr(fp_lib), dstr(metric_lib)))
 
        
        return fp_lib, metric_lib
        
    
    def plot_rlay_mat(self, 
            **kwargs):
        """matrix plot comparing methods for downscaling: rasters
        
        rows: 
            valid, methods
        columns
            depthRaster r2, depthRaster r1, confusionRaster
        """
        
class Plot_samples_wrkr(object):
    def plot_sample_mat(self, 
                        **kwargs):
        """matrix plot comparing methods for downscaling: sampled values
        
        rows: methods
        columns:
            depth histogram, difference histogram, correlation plot
            
        same as Figure 5 on RICorDE paper"""

class PostSession(Plot_rlays_wrkr, ValidateSession):
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
        
    

            
        
        
        
        
 