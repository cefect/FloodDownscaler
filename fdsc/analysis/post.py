'''
Created on Jan. 9, 2023

@author: cefect

data analysis on multiple downscale results
'''


import logging, os, copy, datetime, pickle
import numpy as np
import pandas as pd
import rasterio as rio

from fdsc.analysis.valid import ValidateSession

class PostSession(ValidateSession):
    "Session for analysis on multiple downscale results and their validation metrics"
    def __init__(self, 
                 run_name = None,
                 **kwargs):
 
        if run_name is None:
            run_name = 'post_v1'
        super().__init__(run_name=run_name, **kwargs)
        
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
        res_d = dict()
        data_j =None
        for k, fp in fp_d.items():
            log.debug(f'for {k} loading {ofp}')
            assert os.path.exists(fp)
            
            with open(fp, 'rb') as f:
                data = pickle.load(f)
                assert isinstance(data, dict)
                
                #consistency check
                if not data_j is None:
                    assert set(data_j.keys()).symmetric_difference(data.keys())==set()
                
                
                #store
                res_d[k] = pd.DataFrame.from_dict(data)
                data_j=data
                
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
            
            
        
        
        
        
        
def run_post():
    pass