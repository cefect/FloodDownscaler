'''
Created on Jan. 9, 2023

@author: cefect

data analysis 
'''
from fdsc.analysis.post import PostSession

 

    
 



def aoi08_r32_1215_53(meta_fp_d,
                      run_name='v1',
                      **kwargs):
    
    
    with PostSession(run_name=run_name, **kwargs) as ses:
        
        #load the metadata from teh run
        run_lib, smry_d = ses.load_metas(meta_fp_d)
        
        #get rlays
        rlay_fp_lib = ses.collect_rlay_fps(run_lib)
        
        ses.plot_rlay_mat(rlay_fp_lib)
    


if __name__=='__main__':
    aoi08_r32_1215_53({
        'nodp':r'l:\10_IO\fdsc\outs\ahr_aoi08\ahr_121553_nodp\20230109\ahr_aoi08_ahr_121553_nodp_0109_meta_lib.pkl',
        'cgs':r'l:\10_IO\fdsc\outs\ahr_aoi08\ahr_121553_cgs\20230109\ahr_aoi08_ahr_121553_cgs_0109_meta_lib.pkl'
    })
    #aoi08_r32_1215_53(dryPartial_method='none')
 
    print('done')