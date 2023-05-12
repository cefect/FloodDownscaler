'''
Created on Mar. 28, 2023

@author: cefect

Ahr case study
'''
import os, pickle
from pyproj.crs import CRS
from dscale_2207.pipeline import run_downscale_and_eval 

debug=False


proj_base_dir = r'l:\10_IO\2207_dscale\ins\ahr\aoi13'
def f(fp_rel):
    fp = os.path.join(proj_base_dir, fp_rel)
    assert os.path.exists(fp), fp_rel
    return fp
    
proj_lib = {
    'proj_name': 'ahr_aoi13_0506',
    
    #primary inputs
    'wse2': f(r'fdsc\r32_b10_i65_0511\wd_max_WSE.tif'),
    'dem1': f(r'r04\rim2d\dem005_r04_aoi13_0415.asc'), 
 
    #evaluation
    'inun': f(r'obsv\RLP_LfU_HQ_extrm_WCS_20230324_ahr_4647_aoi13.geojson'), 
    'hwm': f(r'obsv\NWR_ahr11_hwm_20220113b_fix_aoi13.geojson'),
    'wse1':f(r'fdsc\r04_b4_i05_0508\wd_max_WSE.tif'),
    'aoi_fp': r'l:\10_IO\2207_dscale\ins\ahr\aoi13\aoi13_r32_small_0428.geojson',
    'crs':CRS.from_user_input(4647),
    'index_coln':'fid',
    }


#dev
if debug:
    proj_lib['aoi_fp'] = r'l:\10_IO\2207_dscale\ins\ahr\aoi13\aoi09T_0117_4647.geojson'
    run_name='dev'
else:
    run_name='r2'



#===============================================================================
# #build parameters
#===============================================================================
#setup init
init_kwargs = dict(run_name=run_name, relative=False)
for k in ['proj_name','crs', 'aoi_fp', 'index_coln']:
    if k in proj_lib:
        init_kwargs[k]=proj_lib[k]
            

#precompiled results
if debug:
    pick_lib = {
        #=======================================================================
        '0clip':r'L:\10_IO\fdsc\outs\ahr_aoi13_0506\dev\20230510\ahr_aoi13_0506_dev_0510_0clip.pkl',
        '1dsc':r'L:\10_IO\fdsc\outs\ahr_aoi13_0506\dev\20230510\ahr_aoi13_0506_dev_0510_1dsc.pkl',
        # '2eval': 'L:\\10_IO\\fdsc\\outs\\ahr_aoi13_0427\\dev\\20230429\\ahr_aoi13_0427_dev_0429_2eval.pkl' 
        #=======================================================================
        }
else:
    pick_lib={'0clip': 'L:\\10_IO\\fdsc\\outs\\ahr_aoi13_0506\\r2\\20230511\\ahr_aoi13_0506_r2_0511_0clip.pkl',
'1dsc': 'L:\\10_IO\\fdsc\\outs\\ahr_aoi13_0506\\r2\\20230511\\ahr_aoi13_0506_r2_0511_1dsc.pkl',
'2eval': 'L:\\10_IO\\fdsc\\outs\\ahr_aoi13_0506\\r2\\20230511\\ahr_aoi13_0506_r2_0511_2eval.pkl'}

#===============================================================================
# run
#===============================================================================

    

if __name__=='__main__':
    run_downscale_and_eval(proj_lib, pick_lib=pick_lib, **init_kwargs)
    
    print('done')