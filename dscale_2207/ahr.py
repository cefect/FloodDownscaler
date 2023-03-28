'''
Created on Mar. 28, 2023

@author: cefect

Ahr case study
'''
import os, pickle
from pyproj.crs import CRS
from dscale_2207.pipeline import run_downscale, run_eval, run_plot

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
            




#===============================================================================
# run
#===============================================================================
def load_pick(fp):
    with open(fp, 'rb') as f:
        d = pickle.load(f)
    assert isinstance(d, dict)
    print(f'loaded {len(d)} from \n    {fp}')
    return d
    

if __name__=='__main__':
    dsc_res_lib, pick_fp, dsc_vali_res_lib=None, None, None
    
    #precompiled results
    pick_lib = {
        'downscale':r'L:\10_IO\fdsc\outs\ahr_aoi08_0303\dev\20230328\ahr_aoi08_0303_dev_0328_dscM.pkl',
        'eval':r'L:\10_IO\fdsc\outs\ahr_aoi08_0303\dev\20230328\ahr_aoi08_0303_dev_0328_gfps.pkl',
        }
    #===========================================================================
    # downscaling
    #===========================================================================
    k = 'downscale'
    if not k in pick_lib:
        ik = {**{'aoi_fp':proj_lib['aoi']}, **init_kwargs}    
        dsc_res_lib, init_kwargs['logger'] = run_downscale(proj_lib,init_kwargs=ik)
        
    else:
        dsc_res_lib = load_pick(pick_lib[k])

    #===========================================================================
    # evaluation
    #===========================================================================
    k = 'eval'
    if not k in pick_lib: 
        ik = {**{'index_coln':proj_lib['index_coln']}, **init_kwargs}
        vali_kwargs=dict(hwm_pts_fp=proj_lib['hwm'], inun_fp=proj_lib['inun'], aoi_fp=proj_lib['aoi'])
        
        dsc_vali_res_lib, init_kwargs['logger'] = run_eval(dsc_res_lib, 
                                               init_kwargs=ik,vali_kwargs=vali_kwargs)
    else:        
        dsc_vali_res_lib = load_pick(pick_lib[k])
    
 
            
    #===========================================================================
    # plot
    #===========================================================================
    ik = init_kwargs
    
    run_plot(dsc_vali_res_lib, init_kwargs = ik)
    
 
 
    print('done')