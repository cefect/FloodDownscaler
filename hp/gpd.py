'''
Created on Sep. 6, 2022

@author: cefect

geopandas
'''


import geopandas as gpd

from hp.oop import Basic
class GeoPandasWrkr(Basic):
    def __init__(self, 
                 bbox=None,
                 aoi_fp=None,
                 **kwargs):
        
        
        super().__init__(**kwargs)   
        
        
        #=======================================================================
        # bounding box
        #=======================================================================
        if bbox is None:
            
            #load the bounding box from the passed aoi
            if not aoi_fp is None:
                gdf = gpd.read_file(aoi_fp)
                assert len(gdf)==1
                bbox = gdf.geometry.iloc[0]
                
            type(bbox)
        
        self.bbox=bbox
    
    
    