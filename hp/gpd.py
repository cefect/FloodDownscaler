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

import concurrent.futures

#set fiona logging level

logging.getLogger("fiona.collection").setLevel(logging.WARNING)
logging.getLogger("fiona.ogrext").setLevel(logging.WARNING)
logging.getLogger("fiona").setLevel(logging.WARNING)

 

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
    
 
 
def process_coord(c):
    return Point(c)
    
def raster_to_points(rlay_fp, drop_mask=True, max_workers=1):
    """simply convert a raster to a set of points
    
    NOTE: this can be very slow for large rasters
    
    see also hp.rio_to_points for windowed paralleleization
    """
    
    with rio.open(rlay_fp, mode='r') as ds:
        #do some operation
 
        #coordinates
 
        cols, rows = np.meshgrid(np.arange(ds.width), np.arange(ds.height))
 
        xs, ys = rio.transform.xy(ds.transform, rows, cols)
        
        xloc_ar, yloc_ar = np.array(xs), np.array(ys)
        
        #data
 
        ar = ds.read(1, masked=True)
        
        #populate geoseries
 
        coord_l= list(zip(xloc_ar.flatten(), yloc_ar.flatten(), ar.data.flatten()))
        
        """bottleneck here"""
        #=======================================================================
        # plug each coordinate into a point object
        #=======================================================================
        print('GeoSeries %s'%now())
        
        if max_workers==1:
            point_l=[Point(c) for c in coord_l]
        else: #multiprocess 
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                point_l = list(executor.map(process_coord, coord_l))
                
        #=======================================================================
        # collect
        #=======================================================================
        gser_raw = gpd.GeoSeries(point_l,crs=ds.crs)
        
        #handle mask
        if np.any(ar.mask) and drop_mask:
            print('ar.mask.flatten %s'%now()) 
            bx = pd.Series(ar.mask.flatten())
            
            gser = gser_raw.loc[~bx].reset_index(drop=True)
            
        else:
            gser = gser_raw            
        
    gser.name = os.path.basename(rlay_fp)
    return gser
        
        
def drop_z(geo):
    
    assert isinstance(geo, gpd.GeoSeries)
    
    coord_l= list(zip(geo.x.values,geo.y.values))
    
    return gpd.GeoSeries([Point(c) for c in coord_l],
                         index = geo.index, crs=geo.crs, name='geometry')
    
        
 
        
 
 