'''
Created on Mar. 18, 2023

@author: cefect
'''
#===============================================================================
# IMPORTS-------
#===============================================================================
import pytest, os, tempfile, datetime, random
#import rasterio as rio
import shapely.geometry as sgeo
from shapely.geometry import mapping, Polygon
 
#===============================================================================
# import fiona
# import fiona.crs
# from hp.logr import get_new_console_logger, logging
#===============================================================================
from pyproj.crs import CRS


from hp.tests.conftest import temp_dir, logger, out_dir, test_name, init_kwargs
from hp.tests.tools.vectors import generate_random_points

#===============================================================================
# defaults
#===============================================================================
from definitions import epsg,bounds 
crs_default = CRS.from_user_input(epsg)
bbox_default = sgeo.box(*bounds)
    


#===============================================================================
# test data
#===============================================================================
from fperf.tests.data.fred01._main import proj_d as fred_proj_d
proj_lib = dict()
proj_lib['fred01'] = fred_proj_d
 
 

#===============================================================================
# fixtures
#===============================================================================

    
#===============================================================================
# @pytest.fixture(scope='function')
# def test_name(request):
#     return request.node.name.replace('[', '_').replace(']', '_')
# 
# 
# @pytest.fixture(scope='session')
# def logger():
#     return get_new_console_logger(level=logging.DEBUG)
#===============================================================================
    

#===============================================================================
# helpers-------
#===============================================================================
#===============================================================================
# def get_aoi_fp(bbox, crs=crs_default, ofp=None):
#     
#     if ofp is None:
#         ofp = os.path.join(temp_dir, 'aoi.geojson')
#         
#     # write a vectorlayer from a single bounding box
#     assert isinstance(bbox, Polygon)
#     with fiona.open(ofp, 'w', driver='GeoJSON',
#         crs=fiona.crs.from_epsg(crs.to_epsg()),
#         schema={'geometry': 'Polygon',
#                 'properties': {'id':'int'},
#             },
#  
#         ) as c:
#         
#         c.write({ 
#             'geometry':mapping(bbox),
#             'properties':{'id':0},
#             })
#         
#     return ofp
# 
def get_hwm_random_fp(
        count=5,
        colName='water_depth',
        indexColName='id',
        crs=crs_default,
        bbox=bbox_default,
        ofp=None,
        ):
    """build a geojson with random HWM values"""
    #get the random points
    gdf = generate_random_points(bbox.bounds, count).set_crs(crs)
     
    # Add water_depth column with random values between 0 and 2
    gdf[colName] = [random.uniform(0,2) for _ in range(count)]
    gdf.index.name = indexColName
     
    #write
    if ofp is None:
        ofp = os.path.join(temp_dir, 'hwm_test.geojson')
    gdf.to_file(ofp)
    print(f'writing {str(gdf.shape)} to {ofp}')
     
    return ofp
