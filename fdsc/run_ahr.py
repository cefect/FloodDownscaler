'''
Created on Dec. 15, 2022

@author: cefect

downscaling ahr case study
'''

from fdsc.scripts.scripts import run_downscale

def aoi08_r32_1215_53(**kwargs):
    """t=53 for aoi09"""
    return run_downscale(
        r'l:\10_IO\2207_dscale\ins\ahr\aoi08\fdsc\ahr_aoi08_r32_1215-0053_wse.tif',
        r'l:\10_IO\2207_dscale\ins\ahr\aoi08\r04\dem005_r04_aoi08_1210.asc',
        aoi_fp=r'l:\02_WORK\NRC\202110_Ahr\04_CALC\aoi\aoi09_1221_r32.geojson', #target aligned
        **kwargs)
    
    
if __name__=='__main__':
    aoi08_r32_1215_53(dryPartial_method='costDistanceSimple')
    #aoi08_r32_1215_53(dryPartial_method='none')
    print('done')