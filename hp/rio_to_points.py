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