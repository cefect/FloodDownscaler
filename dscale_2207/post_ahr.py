'''
Created on Jan. 9, 2023

@author: cefect

data analysis 
'''
import os
import numpy as np
from fdsc.analysis.post import PostSession

 

#===============================================================================
# setup matplotlib----------
#===============================================================================
cm = 1/2.54

output_format='png'
usetex=False
if usetex:
    os.environ['PATH'] += R";C:\Users\cefect\AppData\Local\Programs\MiKTeX\miktex\bin\x64"
    
  
import matplotlib
#matplotlib.use('Qt5Agg') #sets the backend (case sensitive)
matplotlib.set_loglevel("info") #reduce logging level
import matplotlib.pyplot as plt
 
#set teh styles
plt.style.use('default')
 
#font
matplotlib.rc('font', **{
        'family' : 'serif',
        'weight' : 'normal',
        'size'   : 8})
 
 
for k,v in {
    'axes.titlesize':10,
    'axes.labelsize':10,
    'xtick.labelsize':8,
    'ytick.labelsize':8,
    'figure.titlesize':12,
    'figure.autolayout':False,
    'figure.figsize':(10,10),
    'legend.title_fontsize':'large',
    'text.usetex':usetex,
    }.items():
        matplotlib.rcParams[k] = v
  
print('loaded matplotlib %s'%matplotlib.__version__)
 



def aoi08_r32_1215_53(meta_fp_d,
                      run_name='post_0124',
                      sample_dx_fp=None,
                      **kwargs):
    
    
    with PostSession(run_name=run_name,proj_name='ahr_aoi08', **kwargs) as ses:
        
        #load the metadata from teh run
        run_lib, smry_d = ses.load_metas(meta_fp_d)
        
        #=======================================================================
        # RASTER PLOTS
        #=======================================================================
        #get rlays
        rlay_fp_lib, metric_lib = ses.collect_rlay_fps(run_lib)
        
        #plot them
        ses.plot_rlay_mat(rlay_fp_lib, metric_lib)
         
        #=======================================================================
        # sample metrics
        #=======================================================================
        plt.close()
        df, metric_lib = ses.collect_samples_data(run_lib, sample_dx_fp=sample_dx_fp)
        
        #clear any samples w/ zeros
        bx = np.invert((df==0).any(axis=1))
        df_wet = df.loc[bx, :]
        
        """nodp and cgs are the same if we subset to all wet
        (cgs starts with nodp then expands"""
        ses.plot_samples_mat(df_wet, metric_lib)
    


if __name__=='__main__':
    aoi08_r32_1215_53({
        'nodp':r'l:\10_IO\fdsc\outs\ahr_aoi08\121553_nodp\20230113\ahr_aoi08_121553_nodp_0113_meta_lib.pkl',
        'cgs':r'l:\10_IO\fdsc\outs\ahr_aoi08\121553_cgs\20230113\ahr_aoi08_121553_cgs_0113_meta_lib.pkl',
        'bgl':r'l:\10_IO\fdsc\outs\ahr_aoi08\121553_bgl\20230119\ahr_aoi08_121553_bgl_0119_meta_lib.pkl',
        's14':r'l:\10_IO\fdsc\outs\ahr_aoi08\121553_s14\20230128\ahr_aoi08_121553_s14_0128_meta_lib.pkl',
        },
        #sample_dx_fp=r'L:\10_IO\fdsc\outs\ahr_aoi08\post_0124\20230124\ahr_aoi08_post_0124_0124_collect_samples_data.pkl',
    )
    #aoi08_r32_1215_53(dryPartial_method='none')
 
    print('done')