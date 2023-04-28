'''
Created on Mar. 28, 2023

@author: cefect

Ahr case study
'''
import os, pickle
from pyproj.crs import CRS
from dscale_2207.pipeline import run_downscale, run_eval 

debug=False


proj_base_dir = r'l:\10_IO\2207_dscale\ins\ahr\aoi13'
def f(fp_rel):
    fp = os.path.join(proj_base_dir, fp_rel)
    assert os.path.exists(fp), fp_rel
    return fp
    
proj_lib = {
    'proj_name': 'ahr_aoi13_0427',
    
    #primary inputs
    'wse2': f(r'fdsc\r32_0415_i4\wd_max_WSE.tif'),
    'dem1': f(r'r04\rim2d\dem005_r04_aoi13_0415.asc'), 
 
    #evaluation
    'inun': f(r'obsv\RLP_LfU_HQ_extrm_WCS_20230324_ahr_4647_aoi13.geojson'), 
    'hwm': f(r'obsv\NWR_ahr11_hwm_20220113b_fix_aoi13.geojson'),
    'aoi': r'L:\02_WORK\NRC\202110_Ahr\04_CALC\aoi\aoi13_r64_20230415.geojson',
    'crs':CRS.from_user_input(4647),
    'index_coln':'fid',
    }


#dev
if debug:
    proj_lib['aoi'] = r'l:\10_IO\2207_dscale\ins\ahr\aoi13\aoi09T_0117_4647.geojson'
    run_name='dev'
else:
    run_name='r0'



#===============================================================================
# #build parameters
#===============================================================================
#setup init
init_kwargs = dict(run_name=run_name, relative=False)
for k in ['proj_name','crs']:
    if k in proj_lib:
        init_kwargs[k]=proj_lib[k]
            

#precompiled results
if debug:
    pick_lib = {
 
        'downscale':r'L:\10_IO\fdsc\outs\ahr_aoi13_0427\dev\20230427\ahr_aoi13_0427_dev_0427_dscM.pkl',
        'eval':r'L:\10_IO\fdsc\outs\ahr_aoi13_0427\dev\20230428\ahr_aoi13_0427_dev_0428_rvmd.pkl',
 
        }
else:
    pick_lib={
        #=======================================================================
        # 'downscale':r'l:\10_IO\fdsc\outs\ahr_aoi08_0303\r2\20230328\ahr_aoi08_0303_r2_0328_dscM.pkl',
        # 'eval':r'L:\10_IO\fdsc\outs\ahr_aoi08_0303\r2\20230328\ahr_aoi08_0303_r2_0328_gfps.pkl',
        #=======================================================================
        }
        


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
    #===========================================================================\
    """ moved to standalone script ahr_plot.py"""
 
    print('done')