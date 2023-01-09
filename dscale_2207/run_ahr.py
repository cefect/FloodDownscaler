'''
Created on Dec. 15, 2022

@author: cefect

downscaling ahr case study
'''

from fdsc.analysis.pipeline import run_dsc_vali

def aoi08_r32_1215_53(dryPartial_method='none', 
                      run_name=None,
                      **kwargs):
    
    """t=53 for aoi08
    
    NOTE: we clip  w/ AOI09 to remove some erroneous upstream results in teh validatoin
    """
    
    if run_name is None:
        run_name='ahr_121553_%s'%{'costGrowSimple':'cgs', 'none':'nodp'}[dryPartial_method]
    
    return run_dsc_vali(
        r'l:\10_IO\2207_dscale\ins\ahr\aoi08\fdsc\ahr_aoi08_r32_1215-0053_wse.tif',
        r'l:\10_IO\2207_dscale\ins\ahr\aoi08\r04\dem005_r04_aoi08_1210.asc',
        wse1V_fp=r'c:\LS\10_IO\2207_dscale\ins\ahr\aoi08\fdsc\ahr_aoi08_r04_1215-0053_wse.tif',
        aoi_fp=r'l:\02_WORK\NRC\202110_Ahr\04_CALC\aoi\aoi09_1221_r32.geojson', #target aligned
        
        dsc_kwargs=dict(dryPartial_method=dryPartial_method),
        run_name=run_name,
        **kwargs)
    
    
if __name__=='__main__':
    aoi08_r32_1215_53(dryPartial_method='costGrowSimple')
    #aoi08_r32_1215_53(dryPartial_method='none')
 
    print('done')