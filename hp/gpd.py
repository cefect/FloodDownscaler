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
    
def coord_to_points(coord_l, max_workers=6):
    def f(x):
        return Point(x)
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(f, coord_l))
        
    return results
    
 

def process_window(ds, window):
    # Calculate x, y coordinate arrays for a given window shape
    #rows, cols = np.meshgrid(*map(np.arange, window))
    cols, rows = np.meshgrid(np.arange(window.width), np.arange(window.height))
    
    xs, ys = rio.transform.xy(ds.transform, rows, cols)
    xloc_ar, yloc_ar = np.array(xs), np.array(ys)
    #xs, ys = rio.transform.xy(ds.transform, *np.meshgrid(*map(np.arange, window.shape)))
    
    # Read the data for the given window
    ar = ds.read(1, window=window, masked=True)
    
    # Flatten the x, y, and data arrays, and zip them into a list of tuples
    coord_l = list(zip(xloc_ar.flatten(), yloc_ar.flatten(), ar.data.flatten()))
    
    # Convert each tuple in coord_l into a Point object
    point_l = [Point(c) for c in coord_l]
    
    # Return the list of Point objects
    return point_l

def raster_to_points(rlay_fp, drop_mask=True, max_workers=os.cpu_count()):
    start = now()
    # Open the raster file
    with rio.open(rlay_fp, mode='r') as ds:
        # Generate a list of windows for the raster data
        windows = ds.block_windows(1)
        
        print(f'starting with max_workers={max_workers} and block_shapes={ds.block_shapes}')
        #=======================================================================
        # help(ds.block_windows)
        # 
        # for w in windows:
        #     print(w)
        #=======================================================================
        
        # Initialize an empty list to store the Point objects
        point_l = []
        
        # Use a ThreadPoolExecutor to process each window in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit a task for each window, mapping each future to the corresponding window
            future_to_window = {executor.submit(process_window, ds, window): window for bl, window in windows}
            
            # Iterate over the completed futures, and extend the point_l list with the result of each future
            for future in concurrent.futures.as_completed(future_to_window):
                point_l.extend(future.result())
                
        # Create a GeoSeries from the point_l list and the raster's CRS
        print('assembling geoSeries')
        gser_raw = gpd.GeoSeries(point_l, crs=ds.crs)

        
        #handle mask
        bx = gser_raw.geometry.z==-9999
        if bx.any() and drop_mask:
 
            gser = gser_raw.loc[~bx].reset_index(drop=True)
             
        else:
            gser = gser_raw            
        
    gser.name = os.path.basename(rlay_fp)
    print(f'finished in {now()-start}')
    return gser


def process_coord(c):
    return Point(c)
    
def raster_to_pointsx(rlay_fp, drop_mask=True):
    """convert a raster to a set of points"""
    
    with rio.open(rlay_fp, mode='r') as ds:
        #do some operation
 
        #coordinates
        print('meshgrid %s'%now())
        cols, rows = np.meshgrid(np.arange(ds.width), np.arange(ds.height))
        print('transform %s'%now())
        xs, ys = rio.transform.xy(ds.transform, rows, cols)
        
        xloc_ar, yloc_ar = np.array(xs), np.array(ys)
        
        #data
        print('read %s'%now())
        ar = ds.read(1, masked=True)
        
        #populate geoseries
        print('flatten %s'%now())
        coord_l= list(zip(xloc_ar.flatten(), yloc_ar.flatten(), ar.data.flatten()))
        
        """bottleneck here"""
        print('GeoSeries %s'%now())
        #point_l=[Point(c) for c in coord_l]
        
        """multiprocess"""
        #point_l = coord_to_points(coord_l)
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            point_l = list(executor.map(process_coord, coord_l))
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
    
        
 
        
 
 