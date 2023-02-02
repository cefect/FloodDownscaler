'''
Created on Dec. 15, 2022

@author: cefect

downscaling ahr case study
'''

from fdsc.analysis.pipeline import run_dsc_vali, nicknames_d


def runr(method='none',run_name=None,
          wse2_fp=None, dem1_fp=None,
                      **kwargs):
 
    assert method in nicknames_d, f'method {method} not recognized\n    {list(nicknames_d.keys())}'        
    
    if run_name is None:
        run_name='121553_%s'%nicknames_d[method]
        
    return run_dsc_vali(wse2_fp, dem1_fp,        
        dsc_kwargs=dict(method=method),
        run_name=run_name,  index_coln='fid',
        **kwargs)
    
    

def LM_aoi13_0202(**kwargs):
    
    """
    fv1 solver on v8.1.1. wse max ~1894 event 
    """    
    return runr(
        wse2_fp=r'l:\10_IO\2207_dscale\ins\LM\aoi12\fdsc\lm_aoi12_r128_0202_fv1-0100_wse.tif',
        dem1_fp=r'l:\02_WORK\NRC\2207_dscale\04_CALC\LM\terrain\aoi12\r16\LMFRA_NHC2019_dtm_aoi12_noDikes_r16_f40.tif',
        wse1V_fp=r'l:\10_IO\2207_dscale\ins\LM\aoi12\fdsc\lm_aoi12_r16_0202_fv1-0100_wse.tif',
        sample_pts_fp=r'l:\02_WORK\NRC\2207_dscale\04_CALC\LM\bldgs\microsoft_CanBldgFt_LM_0715_aoi12_pts.geojson',
        aoi_fp=r'l:\04_LIB\02_spatial\LowerFraser\AOI\aoi13_20230202.geojson', 
        proj_name='LM_aoi13',
        **kwargs)

 
    
    
if __name__=='__main__':
    for method in [
        'bufferGrowLoop',
        'costGrowSimple',
        'schumann14', 
        'none',
        ]:
        print(f'\n\nMETHOD={method}\n\n')
        LM_aoi13_0202(method=method)    
 
 
    print('done')