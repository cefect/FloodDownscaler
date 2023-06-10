'''
Created on Jun. 9, 2023

@author: cefect

multi-downscale for volume conservation plots

combining pipeline and case study for now

for 2112_Agg, we used 5 agg resolutions (32, 64, 128, 256, 512) from a base of 10m on the SaintJohn case study

    dsc_l=[1,  2**5, 2**6, 2**7, 2**8, 2**9]
    
for the base analysis, s1=4m and s2=32m (downscale=8)
    let's try: 2, 4, 6, 8
'''

import os, pickle
from pyproj.crs import CRS
from dscale_2207.pipeline import run_downscale_and_eval_multiRes 
 

from dscale_2207.ahr import init_kwargs, proj_lib, debug


#===============================================================================
# parametesr
#===============================================================================


#precompiled results
if debug:
    pick_lib = {
        '0clip':r'L:\10_IO\fdsc\outs\ahr_aoi13_0506\r2\20230609\ahr_aoi13_0506_r2_0609_0clip.pkl',
        '1dems':r'L:\10_IO\fdsc\outs\ahr_aoi13_0506\r2\20230609\ahr_aoi13_0506_r2_0609_1dems.pkl',
        '2dsc':r'L:\10_IO\fdsc\outs\ahr_aoi13_0506\dev\20230609\ahr_aoi13_0506_dev_0609_2dsc.pkl',
        '3wsh':r'L:\10_IO\fdsc\outs\ahr_aoi13_0506\dev\20230610\ahr_aoi13_0506_dev_0610_3wsh.pkl',
        '4stats':'L:\\10_IO\\fdsc\\outs\\ahr_aoi13_0506\\dev\\20230610\\ahr_aoi13_0506_dev_0610_4stats.pkl'
 
        }
    dsc_l=[2**1, 6, 2**3]
else:
    dsc_l=[2**1, 2**2, 6, 2**3]
    pick_lib={'0clip': 'L:\\10_IO\\fdsc\\outs\\ahr_aoi13_0506\\r2\\20230610\\ahr_aoi13_0506_r2_0610_0clip.pkl',
'1dems': 'L:\\10_IO\\fdsc\\outs\\ahr_aoi13_0506\\r2\\20230610\\ahr_aoi13_0506_r2_0610_1dems.pkl',
'2dsc': 'L:\\10_IO\\fdsc\\outs\\ahr_aoi13_0506\\r2\\20230610\\ahr_aoi13_0506_r2_0610_2dsc.pkl',
'3wsh': 'L:\\10_IO\\fdsc\\outs\\ahr_aoi13_0506\\r2\\20230610\\ahr_aoi13_0506_r2_0610_3wsh.pkl',
'4stats': 'L:\\10_IO\\fdsc\\outs\\ahr_aoi13_0506\\r2\\20230610\\ahr_aoi13_0506_r2_0610_4stats.pkl'}

#===============================================================================
# run
#===============================================================================


 

if __name__=='__main__':
    run_downscale_and_eval_multiRes(proj_lib, dsc_l, pick_lib=pick_lib, **init_kwargs)
    
    print('done')
 