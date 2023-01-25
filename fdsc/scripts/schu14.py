'''
Created on Jan. 25, 2023

@author: cefect

Scripts to replicate Schumann 2014's downscaling
'''


import os, datetime, shutil
import numpy as np
import rasterio as rio
from rasterio import shutil as rshutil

from fdsc.scripts.dsc import Dsc_basic

 


class Schuman14(Dsc_basic):
    
    
    def run_schu14(self,wse1_fp, dem_fp,
 
                              **kwargs):
        """run python port of schuman 2014's downscaling"""