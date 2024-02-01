'''
Created on Mar. 21, 2023

@author: cefect

Fredericton test data
'''
import os
from pyproj.crs import CRS
 
src_dir = os.path.dirname(__file__)

proj_d = {
    # test raw data
    'wse2_rlay_fp':os.path.join(src_dir, r'testr_test00_0806_fwse.tif'),
    'aoi_fp':os.path.join(src_dir, r'aoi_T01.geojson'),
    'crs':CRS.from_user_input(3979),
    
    # p1_downscale_wetPartials
    'wse1_rlay2_fp':os.path.join(src_dir, r'wse1_ar2.tif'),
    
    # p2_dp_costGrowSimple._filter_dem_violators
    'wse1_rlay3_fp':os.path.join(src_dir, r'wse1_ar3.tif'),
        
    'dem1_rlay_fp':os.path.join(src_dir, r'dem.tif'),
    
    # validation data
    'wse1_rlayV_fp':os.path.join(src_dir, r'vali/wse1_arV.tif'),
    'sample_pts_fp':os.path.join(src_dir, r'vali/sample_pts_0109.geojson'),
    'samp_gdf_fp':os.path.join(src_dir, r'vali/samps_gdf_0109.pkl'),
    'inun_vlay_fp':os.path.join(src_dir, r'vali/inun_vali1.geojson'),
    'hwm_pts_fp':os.path.join(src_dir, r'vali/hwm_pts_0303.geojson'),
    
 
    }