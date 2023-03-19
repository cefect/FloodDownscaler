'''
Created on Mar. 19, 2023

@author: cefect

tools for helping with vector tests
'''

import geopandas as gpd
from shapely.geometry import Point
import random

def generate_random_points(bbox, n):
    """
    Generate n random points within a bounding box with a water_depth column.

    Parameters
    ----------
    bbox : tuple
        Tuple of (xmin, ymin, xmax, ymax) representing the bounding box.
    n : int
        Number of points to generate.

    Returns
    -------
    GeoDataFrame
        GeoDataFrame of n random points within bbox with water_depth column.
    
    """
    
    xmin, ymin, xmax, ymax = bbox
    points = []
    
    for i in range(n):
        x = random.uniform(xmin,xmax)
        y = random.uniform(ymin,ymax)
        point = Point(x,y)
        points.append(point)

    gdf = gpd.GeoDataFrame(geometry=points)
    
    
    
    return gdf