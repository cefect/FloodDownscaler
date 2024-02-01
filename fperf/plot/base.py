'''
Created on Jan. 9, 2023

@author: cefect

data analysis on multiple downscale results
'''

#===============================================================================
# IMPORTS-----------
#===============================================================================

import logging, os, copy, datetime

from hp.rio_plot import RioPlotr

 
 

 
class PostBase(RioPlotr):
    """base worker"""
    
    def __init__(self, 
                 run_name = None,
                 rowLabels_d=None,
                 **kwargs):
 
        if run_name is None:
            run_name = 'post_v1'
        
        if rowLabels_d is  None:
            rowLabels_d=dict()
        self.rowLabels_d =rowLabels_d
        super().__init__(run_name=run_name, **kwargs)
        

    
    
