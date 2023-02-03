'''
Created on Jan. 9, 2023

@author: cefect

data analysis 
'''
import os
import numpy as np
from fdsc.analysis.post import basic_post_pipeline

 

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
 
 
 
def aoi08_r32_1215_53(**kwargs):
    return basic_post_pipeline(
        {
        'nodp':r'l:\10_IO\fdsc\outs\ahr_aoi08\121553_nodp\20230113\ahr_aoi08_121553_nodp_0113_meta_lib.pkl',
        'cgs':r'l:\10_IO\fdsc\outs\ahr_aoi08\121553_cgs\20230113\ahr_aoi08_121553_cgs_0113_meta_lib.pkl',
        'bgl':r'l:\10_IO\fdsc\outs\ahr_aoi08\121553_bgl\20230119\ahr_aoi08_121553_bgl_0119_meta_lib.pkl',
        's14':r'L:\10_IO\fdsc\outs\ahr_aoi08\121553_s14\20230130\ahr_aoi08_121553_s14_0130_meta_lib.pkl',
        },
        sample_dx_fp=r'L:\10_IO\fdsc\outs\ahr_aoi08\post_0124\20230124\ahr_aoi08_post_0124_0124_collect_samples_data.pkl',   
        run_name='post_0124',proj_name='ahr_aoi08',
        **kwargs)
    
def ahr_aoi08_r32_0130_30(**kwargs):
    return basic_post_pipeline(
        {
        'nodp':r'l:\10_IO\fdsc\outs\ahr_aoi08_0130\121553_nodp\20230130\ahr_aoi08_0130_121553_nodp_0130_meta_lib.pkl',
        'cgs':r'l:\10_IO\fdsc\outs\ahr_aoi08_0130\121553_cgs\20230130\ahr_aoi08_0130_121553_cgs_0130_meta_lib.pkl',
        'bgl':r'l:\10_IO\fdsc\outs\ahr_aoi08_0130\121553_bgl\20230130\ahr_aoi08_0130_121553_bgl_0130_meta_lib.pkl',
        's14':r'l:\10_IO\fdsc\outs\ahr_aoi08_0130\121553_s14\20230130\ahr_aoi08_0130_121553_s14_0130_meta_lib.pkl',
        },
        #sample_dx_fp=r'L:\10_IO\fdsc\outs\ahr_aoi08_0130\p0130\20230130\ahr_aoi08_0130_p0130_0130_collect_samples_data.pkl',   
        run_name='p0130',proj_name='ahr_aoi08_0130',
        **kwargs)
    
    
if __name__=='__main__':
    ahr_aoi08_r32_0130_30()
   
 
    print('done')