'''
Created on Dec. 15, 2022

@author: cefect

downscaling LowerMainland case study
'''
 
from fdsc.analysis.pipeline import run_pipeline_multi, nicknames_d


def runr(**kwargs):        
    return run_pipeline_multi(index_coln='fid',
                              **kwargs)
    
    

def LM_aoi13_0202(**kwargs):
    
    """
    fv1 solver on v8.1.1. wse max ~1894 event 
    """    
    return runr(
        wse2_fp=r'l:\10_IO\2207_dscale\ins\LM\aoi12\fdsc\lm_aoi12_r128_0202_fv1-0100_wse.tif',
        dem1_fp=r'l:\02_WORK\NRC\2207_dscale\04_CALC\LM\terrain\aoi12\r16\LMFRA_NHC2019_dtm_aoi12_noDikes_r16.tif', #unmasked
        wse1V_fp=r'l:\10_IO\2207_dscale\ins\LM\aoi12\fdsc\lm_aoi12_r16_0202_fv1-0100_wse.tif',
        sample_pts_fp=r'l:\02_WORK\NRC\2207_dscale\04_CALC\LM\bldgs\microsoft_CanBldgFt_LM_0715_aoi12_pts.geojson',
        aoi_fp=r'l:\04_LIB\02_spatial\LowerFraser\AOI\aoi13_20230202.geojson', 
        proj_name='LM_aoi13',
        **kwargs)

 
def LM_aoi13_0203(**kwargs):
    
    """
    fv1 solver on v8.1.1. wse max ~1894 event 
    
    r128 @ n=
    """    
    return runr(
        wse2_fp=r'l:\10_IO\2207_dscale\ins\LM\aoi12\fdsc\lm_aoi12_r128_0202q_0.0225-0100_wse.tif',
        dem1_fp=r'l:\02_WORK\NRC\2207_dscale\04_CALC\LM\terrain\aoi12\r16\LMFRA_NHC2019_dtm_aoi12_noDikes_r16.tif', #unmasked
        wse1V_fp=r'l:\10_IO\2207_dscale\ins\LM\aoi12\fdsc\lm_aoi12_r16_0202q-0100_wse.tif',
        sample_pts_fp=r'l:\02_WORK\NRC\2207_dscale\04_CALC\LM\bldgs\microsoft_CanBldgFt_LM_0715_aoi12_pts.geojson',
        aoi_fp=r'l:\04_LIB\02_spatial\LowerFraser\AOI\aoi13_20230202.geojson', 
        proj_name='LM_aoi13',
        **kwargs)
    
if __name__=='__main__':
    LM_aoi13_0203()