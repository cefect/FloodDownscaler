'''
Created on Jan. 9, 2023

@author: cefect

data analysis on multiple downscale results
'''


import logging, os, copy, datetime, pickle, pprint, math
 
from rasterio.plot import show

import matplotlib.pyplot as plt
import matplotlib
 

 


 
 
from fperf.plot.base import PostBase

 
 
class Fdsc_Plot_Base(PostBase):
    """base worker"""
    
    """not used... was also on fperf.plot.pipeline.PostSession()
    sim_color_d = {'CostGrow': '#e41a1c','Basic': '#377eb8','SimpleFilter': '#984ea3','Schumann14': '#ffff33', 'WSE2': '#f781bf', 'WSE1': '#999999'}
    """
    
    #confusion_color_d = {'FN':'#c700fe', 'FP':'red', 'TP':'#00fe19', 'TN':'white'}
    
    #nicknames_d = nicknames_d.copy()
    
    #rowLabels_d = {'WSE1':'Hydrodyn. (s1)', 'WSE2':'Hydrodyn. (s2)'}
    
    def __init__(self, 
                 run_name = None,
                 **kwargs):
 
        if run_name is None:
            run_name = 'post_v1'
        super().__init__(run_name=run_name, **kwargs)
    
    

    
       

