'''
Created on Dec. 15, 2022

@author: cefect

downscaling ahr case study
'''
from fdsc.analysis.pipeline import run_pipeline_multi, nicknames_d


def runr(**kwargs):        
    return run_pipeline_multi(index_coln='fid',
                              **kwargs)
    
    

#===============================================================================
# def aoi08_r32_1215_53(**kwargs):
#     
#     """t=53 for aoi08
#     
#     NOTE: we clip  w/ AOI09 to remove some erroneous upstream results in teh validatoin
#     """    
#     return runr(
#         wse2_fp=r'l:\10_IO\2207_dscale\ins\ahr\aoi08\fdsc\ahr_aoi08_r32_1215-0053_wse.tif',
#         dem1_fp=r'l:\10_IO\2207_dscale\ins\ahr\aoi08\r04\dem005_r04_aoi08_1210.asc',
#         wse1V_fp=r'l:\10_IO\2207_dscale\ins\ahr\aoi08\fdsc\ahr_aoi08_r04_1215-0053_wse.tif',
#         sample_pts_fp=r'l:\10_IO\2207_dscale\ins\ahr\aoi08\bldgs\osm_buildings_aoi07_1114_poly_a50_cent_aoi09.geojson',
#         aoi_fp=r'l:\02_WORK\NRC\202110_Ahr\04_CALC\aoi\aoi09_1221_r32.geojson', 
#         proj_name='ahr_aoi08',
#         **kwargs)
#===============================================================================

def ahr_aoi08_r32_0130_30(**kwargs):
    return runr(proj_name='ahr_aoi08_0130',
        wse2_fp=r'l:\10_IO\2207_dscale\ins\ahr\aoi08\fdsc\ahr_aoi08_r32_0130_30\ahr_aoi08_r32_1221-0030_wse.tif',
        dem1_fp=r'l:\10_IO\2207_dscale\ins\ahr\aoi08\r04\dem005_r04_aoi08_1210.asc',
        wse1V_fp=r'l:\10_IO\2207_dscale\ins\ahr\aoi08\fdsc\ahr_aoi08_r32_0130_30\ahr_aoi08_r04_1215-0030_wse.tif',
        inun_fp=r'L:\02_WORK\NRC\202110_Ahr\01_GEN\01_INOUT\2022 01 13 - Heiko - 2021 data\Ahr_Überflutungsflächen_HW07-2021_LfU712_2021_07_14.shp', 
        sample_pts_fp=r'l:\10_IO\2207_dscale\ins\ahr\aoi08\bldgs\osm_buildings_aoi07_1114_poly_a50_cent_aoi09.geojson',
        aoi_fp=r'l:\02_WORK\NRC\202110_Ahr\04_CALC\aoi\aoi09_1221_r32.geojson',        
        **kwargs)
    
def ahr11_rim0201_r32_0203(**kwargs):
    """Heiko's RIM2D Feb 2023 run.
    r32 is low
    r4 has houses removed... so we get lots of FP assets
        use buffered polygons instead?
        ask for a run with r04 houses burned?
        use r32 w/ houses burned"""
    return runr(proj_name='ahr11_0203',
        wse2_fp=r'l:\10_IO\2207_dscale\ins\ahr\aoi11\fdsc\Altenahr-Sinzig_r32_flood2021_nohouses_max_wse_0204.tif',
        dem1_fp=r'l:\02_WORK\NRC\2207_dscale\04_CALC\ahr\terrain\aoi11_0129\r04\dem005_r04_aoi11_0129.tif',
        wse1V_fp=r'l:\10_IO\2207_dscale\ins\ahr\aoi11\fdsc\Altenahr-Sinzig_r04_flood2021_reconstructed_max_wse_0203.tif',
        sample_pts_fp=r'l:\02_WORK\NRC\2207_dscale\04_CALC\ahr\bldgs\osm_buildings_aoi07_1114_poly_a50_c240_pts.geojson',
        aoi_fp=r'L:\02_WORK\NRC\202110_Ahr\04_CALC\aoi\aoi11b_20230204.geojson',        
        **kwargs)
    
if __name__=='__main__':
    ahr_aoi08_r32_0130_30(method_l=[
                    #'bufferGrowLoop',
                    'CostGrow',
                    'Basic', 
                    'SimpleFilter',
                    'Schumann14',
                    ])
    
    #===========================================================================
    # ahr11_rim0201_r32_0203(method_l=[
    #                 #'bufferGrowLoop',
    #                 'CostGrow',
    #                 'Schumann14', 
    #                 'none',
    #                 'SimpleFilter',
    #                 ]) 
    #===========================================================================
 
 
    print('done')