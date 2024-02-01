'''
Created on Feb. 4, 2023

@author: cefect

convert raster pixels to points
    more sophisitcted windowed parallelization
    see also hp.gpd.raster_to_points
    
'''
import os, logging, datetime
import rasterio as rio
import numpy as np
from shapely.geometry import Point

import concurrent.futures
import geopandas as gpd
from hp.gpd import set_mask

def now():
    return datetime.datetime.now()


def process_window(ds, window):
    
    # Calculate x, y coordinate arrays for a given window shape
    #rows, cols = np.meshgrid(*map(np.arange, window))
    cols, rows = np.meshgrid(np.arange(window.width), np.arange(window.height))
    
    xs, ys = rio.transform.xy(ds.transform, rows, cols)
    xloc_ar, yloc_ar = np.array(xs), np.array(ys)
    #xs, ys = rio.transform.xy(ds.transform, *np.meshgrid(*map(np.arange, window.shape)))
    
    # Read the data for the given window
    ar = ds.read(1, window=window, masked=True)
    
    assert ar.fill_value==-9999
    
    # Flatten the x, y, and data arrays, and zip them into a list of tuples
    
    coord_l = list(zip(xloc_ar.flatten(), yloc_ar.flatten(), ar.filled().flatten()))
    
    # Convert each tuple in coord_l into a Point object
    point_l = [Point(c) for c in coord_l]
    
    # Return the list of Point objects
    return point_l




def raster_to_points_windowed(rlay_fp, drop_mask=False, max_workers=os.cpu_count()):
    """convert raster pixels to point using window paralleization
    
    PERFORMANCE TESTS
    ---------------
    see hp.tests.text_pix_to_points
    """
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

        
        gser = set_mask(gser_raw, drop_mask)            
        
    gser.name = os.path.basename(rlay_fp)
    print(f'finished in {now()-start}')
    return gser


def process_coord(c):
    return Point(c)
    
def raster_to_points_simple(rlay_fp, drop_mask=True, max_workers=1):
    """simply convert a raster to a set of points
    
    NOTE: this can be very slow for large rasters
    
    see also hp.rio_to_points for windowed paralleleization
    
    PERFORMANCE TESTS
    ---------------
    max_workers>1 slows things down tremendously.
        GeoRaster package works much better. see hp.gr.pixels_to_points
     
    """
    if max_workers is None:
        max_workers=os.cpu_count()
    
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
        print(f'preparing GeoSeries on {ar.shape} w/ max_workers={max_workers} %s'%now())
        
        if max_workers==1:
            point_l=[Point(c) for c in coord_l]
        else: #multiprocess 
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                point_l = list(executor.map(process_coord, coord_l))
                
        #=======================================================================
        # collect
        #=======================================================================
        print(f'collecting geoseries on {len(point_l)} %s'%now())
        gser_raw = gpd.GeoSeries(point_l,crs=ds.crs)
        
        #handle mask
        gser = set_mask(gser_raw, drop_mask)            
        
    gser.name = os.path.basename(rlay_fp)
    return gser





