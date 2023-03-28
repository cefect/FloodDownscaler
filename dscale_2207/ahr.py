'''
Created on Mar. 28, 2023

@author: cefect

Ahr case study
'''
import os, pickle
from pyproj.crs import CRS
from dscale_2207.pipeline import run_downscale, run_eval

debug=True


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
    'inun': f(r'inun\inun_anschlaglinie_HW_7_21_220223_aoi09_0303.geojson'),
    'sample_pts_fp': f(r'bldgs\osm_buildings_aoi07_1114_poly_a50_cent_aoi09.geojson'),
    'hwm': f(r'hwm\NWR_ahr11_hwm_20220113b_fix_aoi09.geojson'),
    'aoi': f('aoi09_1221_r32.geojson'),
    'crs':CRS.from_user_input(25832),
    'index_coln':'fid',
    }


#dev
if debug:
    proj_lib['aoi'] = f('aoi09T_0117.geojson')
    run_name='dev'
else:
    run_name='r2'



#===============================================================================
# #build parameters
#===============================================================================
#setup init
init_kwargs = dict(run_name=run_name, relative=False)
for k in ['proj_name','crs']:
    if k in proj_lib:
        init_kwargs[k]=proj_lib[k]
            


vali_kwargs=dict(hwm_pts_fp=proj_lib['hwm'], inun_fp=proj_lib['inun'], aoi_fp=proj_lib['aoi'])

#===============================================================================
# run
#===============================================================================
if __name__=='__main__':
 
    #===========================================================================
    # downscaling
    #===========================================================================
    ik = {**{'aoi_fp':proj_lib['aoi']}, **init_kwargs}
    dsc_res_lib, pick_fp=None, None
 
    
    dsc_res_lib, init_kwargs['logger'] = run_downscale(proj_lib,init_kwargs=ik)
    
    #pick_fp=r'L:\10_IO\fdsc\outs\ahr_aoi08_0303\dev\20230328\ahr_aoi08_0303_dev_0328_dscM.pkl'
    
    #load run_downscale results library
    if dsc_res_lib is None:
        with open(pick_fp, 'rb') as f:
            dsc_res_lib = pickle.load(f)
    #===========================================================================
    # evaluation
    #===========================================================================
 
    ik = {**{'index_coln':proj_lib['index_coln']}, **init_kwargs}
    run_eval(dsc_res_lib, init_kwargs=ik, vali_kwargs=vali_kwargs)
 
 
    print('done')