'''
Created on Mar. 8, 2023

@author: cefect
'''

import fiona #not a rasterio dependency? needed for aoi work
from pyproj.crs import CRS
import shapely.geometry as sgeo


def get_bbox_and_crs(fp):
    with fiona.open(fp, "r") as source:
        bbox = sgeo.box(*source.bounds) 
        crs = CRS(source.crs['init'])
        
    return bbox, crs