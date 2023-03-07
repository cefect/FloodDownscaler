'''
Created on Sep. 6, 2022

@author: cefect

geopandas
'''

 
import shapely, os, logging, datetime
 
import shapely.geometry as sgeo
import numpy as np
import pandas as pd
from shapely.geometry import Point, polygon
import rasterio as rio
from pyproj.crs import CRS

import geopandas as gpd

 
#set fiona logging level

logging.getLogger("fiona.collection").setLevel(logging.WARNING)
logging.getLogger("fiona.ogrext").setLevel(logging.WARNING)
logging.getLogger("fiona").setLevel(logging.WARNING)

 
from hp.oop import Basic
 
 

def now():
    return datetime.datetime.now()
 
 
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
    
    
    
def get_samples(gser, rlay_ds, colName=None):
    assert isinstance(gser, gpd.geoseries.GeoSeries)
    assert np.all(gser.geom_type=='Point')
    assert isinstance(rlay_ds, rio.io.DatasetReader), type(rlay_ds)
    if colName is None: colName = os.path.basename(rlay_ds.name)
    
    #get points
    coord_l = [(x,y) for x,y in zip(gser.x , gser.y)]
    samp_l = [x[0] for x in rlay_ds.sample(coord_l)]
 
    
    #replace nulls
    samp_ar = np.where(np.array([samp_l])==rlay_ds.nodata, np.nan, np.array([samp_l]))[0]
    
    
    
    return gpd.GeoDataFrame(data={colName:samp_ar}, index=gser.index, geometry=gser)
    
 
 
  
def drop_z(geo):
    
    assert isinstance(geo, gpd.GeoSeries)
    
    coord_l= list(zip(geo.x.values,geo.y.values))
    
    return gpd.GeoSeries([Point(c) for c in coord_l],
                         index = geo.index, crs=geo.crs, name='geometry')
    
        
def set_mask(gser_raw, drop_mask):
    assert gser_raw.geometry.z.notna().all(), 'got some bad z values'
    #handle mask
    bx = gser_raw.geometry.z == -9999
    if bx.any() and drop_mask:
        gser = gser_raw.loc[~bx].reset_index(drop=True)
    else:
        gser = gser_raw
    return gser
        
 
 