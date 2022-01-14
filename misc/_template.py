'''
Created on Mar. 27, 2021

@author: cefect
'''
#===============================================================================
# imports-----------
#===============================================================================
import os, datetime
import pandas as pd
import numpy as np

start =  datetime.datetime.now()
print('start at %s'%start)
today_str = datetime.datetime.today().strftime('%Y%m%d')


from hp.exceptions import Error
from hp.dirz import force_open_dir
from hp.oop import Basic
from hp.plot import Plotr #only needed for plotting sessions
from hp.Q import Qproj #only for Qgis sessions
 


#===============================================================================
# vars
#===============================================================================
work_dir = os.path.dirname(os.path.dirname(__file__))


#===============================================================================
# CLASSES----------
#===============================================================================
class Session(Basic):
    def __init__(self, **kwargs):
        
        super().__init__(work_dir=work_dir, #out_dir=os.path.join(work_dir, 'out'),
                         tag='job', **kwargs)
    

class SessionPlot(Plotr):
    
    
    def __init__(self,
                 **kwargs):
    
        super().__init__(work_dir=work_dir, #out_dir=os.path.join(work_dir, 'out'),
                         figsize=(14,5),
                         tag='job',
                          **kwargs) #initilzie teh baseclass
        
        self._init_plt()
        
        
class SessionQ(Qproj):
    
    def __init__(self, **kwargs):
        
        super().__init__(work_dir=work_dir, #out_dir=os.path.join(work_dir, 'out'),
                         tag='job', **kwargs)
        






#===============================================================================
# FUNCTIONS-----------------
#===============================================================================






















if __name__ =="__main__": 
    
    Session()
    
    tdelta = datetime.datetime.now() - start
    print('finished in %s'%tdelta)
