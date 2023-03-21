'''
Created on Mar. 19, 2023

@author: cefect

tools for helping with vector tests
'''
import os, tempfile, datetime
import fiona
import fiona.crs
import geopandas as gpd
import shapely.geometry as sgeo
from shapely.geometry import Point, mapping, Polygon
from pyproj.crs import CRS
 
import random

from hp.tests.conftest import temp_dir

#spatial defaults
from definitions import epsg,bounds 
crs_default = CRS.from_user_input(epsg)
bbox_default = sgeo.box(*bounds)

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


def get_aoi_fp(bbox, crs=crs_default, ofp=None, out_dir=None,):
    
    if ofp is None:
        if out_dir is None: out_dir=temp_dir
        ofp = os.path.join(out_dir, 'aoi.geojson')
        
    # write a vectorlayer from a single bounding box
    assert isinstance(bbox, Polygon)
    with fiona.open(ofp, 'w', driver='GeoJSON',
        crs=fiona.crs.from_epsg(crs.to_epsg()),
        schema={'geometry': 'Polygon',
                'properties': {'id':'int'},
            },
 
        ) as c:
        
        c.write({ 
            'geometry':mapping(bbox),
            'properties':{'id':0},
            })
        
    return ofp