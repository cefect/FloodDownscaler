'''
Created on Jan. 9, 2023

@author: cefect

data analysis 
'''
from fdsc.analysis.post import PostSession

def plot_rlay_mat(
        meta_fp_d,
        run_name='v1',
        **kwargs):
    """matrix plot comparing methods for downscaling: rasters
    
    rows: methods
    columns
        depthRaster, confusionRaster
    """
    
    
    with PostSession(run_name=run_name, **kwargs) as ses:
        
        #load the metadata from teh run
        run_lib = ses.load_metas(meta_fp_d)
    
 
def plot_sample_mat():
    """matrix plot comparing methods for downscaling: sampled values
    
    rows: methods
    columns:
        depth histogram, difference histogram, correlation plot
        
    same as Figure 5 on RICorDE paper"""





if __name__=='__main__':
    aoi08_r32_1215_53({
        'nodp':r'l:\10_IO\fdsc\outs\ahr_aoi08\ahr_121553_nodp\20230109\ahr_aoi08_ahr_121553_nodp_0109_meta_lib.pkl',
        'cgs':r'l:\10_IO\fdsc\outs\ahr_aoi08\ahr_121553_cgs\20230109\ahr_aoi08_ahr_121553_cgs_0109_meta_lib.pkl'
    })
    #aoi08_r32_1215_53(dryPartial_method='none')
 
    print('done')