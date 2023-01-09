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
        
        
        
def run_post():
    pass