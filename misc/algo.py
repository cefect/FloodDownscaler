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
 
        
        
class Session(Qproj):
    
    def __init__(self, **kwargs):
        
        super().__init__(work_dir=work_dir, 
                         tag='fixgeo', **kwargs)


if __name__ =="__main__": 
    
    fp = r'C:\LS\03_TOOLS\misc\outs\20210517\EGS_Flood_Product_Archive_20210329_sel.gpkg'
    
    _, fn = os.path.split(fp)
    fni, ext = os.path.splitext(fn)
    
    wrkr = Session()
    
    wrkr.fixgeo(fp, output=os.path.join(wrkr.out_dir, '%s_fix.gpkg'%fni))
    
    
    
    tdelta = datetime.datetime.now() - start
    print('finished in %s'%tdelta)
