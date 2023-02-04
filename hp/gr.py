'''
Created on Feb. 4, 2023

@author: cefect

georasters
'''
import georasters as gr
import geopandas as gpd
from hp.gdal import getCrs

def pixels_to_points(fp):
    """convert raster pixels to points using GeoRaster"""
    data_gr = gr.from_file(fp)
    gr.get_geo_info(fp)
    data_df = data_gr.to_pandas()
    crs =   getCrs(fp)
    gser = gpd.GeoSeries(gpd.points_from_xy(data_df.x,data_df.y, data_df.iloc[:, 2]),crs=crs)
 
    return gser
    