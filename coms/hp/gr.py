'''
Created on Feb. 4, 2023

@author: cefect

georasters
'''
import georasters as gr
import geopandas as gpd
from hp.gdal import getCrs
from hp.gpd import set_mask, view


def pixels_to_points(fp):
    """convert raster pixels to points using GeoRaster"""
    data_gr = gr.from_file(fp)
     
    """WARNING... this drops nulls
    giving corners?"""
    #data_df = data_gr.to_pandas()
    #===========================================================================
    # crs =   getCrs(fp)
    # return gpd.GeoSeries(gpd.points_from_xy(data_df.x,data_df.y, data_df.iloc[:, 2]),crs=crs)
    #===========================================================================
    
    gdf_raw = data_gr.to_geopandas()
    
    grid_centers = gdf_raw.centroid
    
 
    return gpd.GeoSeries(
                gpd.points_from_xy(grid_centers.x, grid_centers.y, z=gdf_raw['value']),
                crs=gdf_raw.crs)
    
    """
    view(data_gr.to_geopandas())
    """
    
    

 