'''
Created on Dec. 15, 2022

@author: cefect

downscaling ahr case study
'''

from fdsc.analysis.pipeline import run_dsc_vali, nicknames_d

def aoi08_r32_1215_53(method='none', 
                      run_name=None,
                      **kwargs):
    
    """t=53 for aoi08
    
    NOTE: we clip  w/ AOI09 to remove some erroneous upstream results in teh validatoin
    """
 
    assert method in nicknames_d, f'method {method} not recognized\n    {list(nicknames_d.keys())}'        
    
    if run_name is None:
        run_name='121553_%s'%nicknames_d[method]
    
    return run_dsc_vali(
        r'l:\10_IO\2207_dscale\ins\ahr\aoi08\fdsc\ahr_aoi08_r32_1215-0053_wse.tif',
        r'l:\10_IO\2207_dscale\ins\ahr\aoi08\r04\dem005_r04_aoi08_1210.asc',
        wse1V_fp=r'l:\10_IO\2207_dscale\ins\ahr\aoi08\fdsc\ahr_aoi08_r04_1215-0053_wse.tif',
        sample_pts_fp=r'l:\10_IO\2207_dscale\ins\ahr\aoi08\bldgs\osm_buildings_aoi07_1114_poly_a50_cent_aoi09.geojson',
        aoi_fp=r'l:\02_WORK\NRC\202110_Ahr\04_CALC\aoi\aoi09_1221_r32.geojson', #target aligned
        
        dsc_kwargs=dict(method=method),
        run_name=run_name, proj_name='ahr_aoi08', index_coln='fid',
        **kwargs)
    
    
if __name__=='__main__':
    #aoi08_r32_1215_53(method='bufferGrowLoop')
    #aoi08_r32_1215_53(method='costGrowSimple')
    aoi08_r32_1215_53(method='schumann14')
    #aoi08_r32_1215_53(method='none')
 
    print('done')