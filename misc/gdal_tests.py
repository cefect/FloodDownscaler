'''
Created on May 22, 2021

@author: cefect
'''




 

#===============================================================================
# imports-----------
#===============================================================================
import os, datetime, sys
import pandas as pd
import numpy as np

start =  datetime.datetime.now()
print('start at %s'%start)
today_str = datetime.datetime.today().strftime('%Y%m%d')

#===============================================================================
# gdals
#===============================================================================
from osgeo import gdal
from osgeo import ogr
from osgeo import osr
from osgeo import gdal_array
from osgeo import gdalconst

#add the gald scrillpts location
sys.path.append(r'C:\OSGeo4W64\apps\Python37\Scripts')
 
 
from gdal_merge import main
#===============================================================================
# customs
#===============================================================================

from hp.exceptions import Error
from hp.dirz import force_open_dir
 
from hp.Q import Qproj #only for Qgis sessions
 


#===============================================================================
# vars
#===============================================================================
work_dir = os.path.dirname(os.path.dirname(__file__))

class Session(Qproj):
    
    def __init__(self, **kwargs):
        
        super().__init__(work_dir=work_dir, out_dir=os.path.join(work_dir, 'out'),
                         tag='job', **kwargs)






if __name__ =="__main__": 
    
    #===========================================================================
    # help(gdal.Rasterize)
    # help(gdal.RasterizeOptions)
    #===========================================================================
    
    #===========================================================================
    # native python api
    #===========================================================================
    #===========================================================================
    # opts = gdal.RasterizeOptions(
    #     layers = ['aoi03_fred_20210325'],
    #     burnValues=[1.0],
    #     xRes=10, yRes=10,
    #     noData=0.0,
    #     #outputType='Float32',
    #     format='GTiff',        
    #     )
    #     
    # dest=r'C:\LS\03_TOOLS\_jobs\202103_InsCrve\_ins\0522\r1.tif'
    # src = r'C:/LS/02_WORK/02_Mscripts/InsuranceCurves/04_CALC/Fred/aoi/aoi03_fred_20210325.gpkg'
    # gdal.Rasterize(dest, src, options=opts)
    #===========================================================================
    
    #run gdal_rasterize from a python script
    import subprocess
    cmd_str= r"gdal_rasterize -l aoi03_fred_20210325 -burn 1.0 -tr 10.0 10.0 -a_nodata 0.0 -te 2146359.6353 144064.6973 2162963.1236 152171.7172 -ot Float32 -of GTiff C:/LS/02_WORK/02_Mscripts/InsuranceCurves/04_CALC/Fred/aoi/aoi03_fred_20210325.gpkg C:\LS\03_TOOLS\_jobs\202103_InsCrve\_ins\0522\r1.tif"
    print('running \n%s'%cmd_str)
    subprocess.run(cmd_str)
    #subprocess.Popen('notepad.exe')
    #===========================================================================
    # args = ['gdal_rasterize', 'l']
    # subprocess.run(['C:\\Temp\\a b c\\Notepad.exe', 'C:\\test.txt'])
    #===========================================================================

    
    "gdal_rasterize -l aoi03_fred_20210325 -burn 1.0 -tr 10.0 10.0 -a_nodata 0.0 -te 2146359.6353 144064.6973 2162963.1236 152171.7172 -ot Float32 -of GTiff C:/LS/02_WORK/02_Mscripts/InsuranceCurves/04_CALC/Fred/aoi/aoi03_fred_20210325.gpkg C:\LS\03_TOOLS\_jobs\202103_InsCrve\_ins\0522\r1.tif"
    
    tdelta = datetime.datetime.now() - start
    print('finished in %s'%tdelta)