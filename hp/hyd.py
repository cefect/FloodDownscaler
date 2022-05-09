'''
Created on May 9, 2022

@author: cefect

flood specific Q scripts
'''
import os, datetime
from hp.Q import Qproj


class HQproj(Qproj):
    def __init__(self, 
                 dem_fp=None,
                 **kwargs):
 
        super().__init__(  **kwargs) 
        
        if not dem_fp is None:
            self.load_dem(fp=dem_fp)
            
    def load_dem(self,
                 fp=None
                 ):
        
        assert os.path.exists(fp)
    
