'''
Created on Sep. 6, 2022

@author: cefect

geopandas
'''

import shapely
import shapely.geometry as sgeo
from shapely.geometry import polygon
import rasterio as rio
from pyproj.crs import CRS

import geopandas as gpd

from hp.oop import Basic
class GeoPandasWrkr(object):
    def __init__(self, 
                 bbox=None,
                 aoi_fp=None,
                 crs=CRS.from_user_input(4326),
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
 
        if not bbox is None:                
            assert isinstance(bbox, polygon.Polygon), type(bbox)
 
        
        self.bbox=bbox
        
        #=======================================================================
        # crs
        #=======================================================================
        if not crs is None:
            assert isinstance(crs, CRS), type(crs)
            
        self.crs=crs
        
#===============================================================================
# def ds_get_bounds(ds):
#     b =ds.bounds
#     return sgeo.box(b.left, b.right, b.top, b.bottom)
#===============================================================================

def get_multi_intersection(poly_l):
    """compute the intersection of many shapely polygons
    surprised there is no builtin
    """
    
    res = None
    for poly in poly_l:
        if poly is None: continue
        if res is None: 
            res=poly
            continue
        assert isinstance(poly, sgeo.polygon.Polygon)
        assert res.intersects(poly)
        res = res.intersection(poly)
        
    assert res.area>0
    
    return res
    
    
    
#===============================================================================
# def assert_intersect(bounds_left,bounds_right, msg='',): 
#     """check if objects intersect"""
#     if not __debug__: # true if Python was not started with an -O option
#         return
#     
#     __tracebackhide__ = True 
#     
#     assert rio.coords.disjoint_bounds(bounds_left, bounds_right), 'disjoint'
#===============================================================================
    
    