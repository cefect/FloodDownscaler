'''
Created on Mar. 28, 2023

@author: cefect

Ahr case study
'''
import os
from pyproj.crs import CRS
from dscale_2207.pipeline import run_downscale


proj_base_dir = r'l:\10_IO\2207_dscale\ins\ahr\aoi08'
def f(fp_rel):
    fp = os.path.join(proj_base_dir, fp_rel)
    assert os.path.exists(fp), fp_rel
    return fp
    
proj_lib = {
    'proj_name': 'ahr_aoi08_0303',
    'wse2': f(r'fdsc\ahr_aoi08_r32_0130_30\ahr_aoi08_r32_1221-0030_wse.tif'),
    'dem1': f(r'r04\dem005_r04_aoi08_1210.tif'), 
    'true_wse_fp': f(r'fdsc\ahr_aoi08_r32_0130_30\ahr_aoi08_r04_1215-0030_wse.tif'),
    'true_inun_fp': f(r'inun\inun_anschlaglinie_HW_7_21_220223_aoi09_0303.geojson'),
    'sample_pts_fp': f(r'bldgs\osm_buildings_aoi07_1114_poly_a50_cent_aoi09.geojson'),
    'hwm_pts_fp': f(r'hwm\NWR_ahr11_hwm_20220113b_fix_aoi09.geojson'),
    'aoi': f('aoi09_1221_r32.geojson'),
    'crs':CRS.from_user_input(25832)
    }


proj_lib_dev = proj_lib.copy()
proj_lib_dev['aoi'] = f('aoi09T_0117.geojson')

if __name__=='__main__':
    
    #run_downscale(proj_lib,run_name='r2')
    run_downscale(proj_lib_dev,init_kwargs=dict(run_name='dev'))
 
 
    print('done')