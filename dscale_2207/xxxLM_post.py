'''
Created on Jan. 9, 2023

@author: cefect

data analysis 
'''
import os
import numpy as np
from fdsc.analysis.post import PostSession, basic_post_pipeline

 

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
  
print('loaded matplotlib %s' % matplotlib.__version__)
    

def LM_aoi13_0203(**kwargs):
    return basic_post_pipeline(
            {'cgs': 'L:\\10_IO\\fdsc\\outs\\LM_aoi13\\cgs\\20230203\\LM_aoi13_cgs_0203_meta_lib.pkl',
         's14': 'L:\\10_IO\\fdsc\\outs\\LM_aoi13\\s14\\20230203\\LM_aoi13_s14_0203_meta_lib.pkl',
         'none': 'L:\\10_IO\\fdsc\\outs\\LM_aoi13\\none\\20230203\\LM_aoi13_none_0203_meta_lib.pkl',
         'nodp': 'L:\\10_IO\\fdsc\\outs\\LM_aoi13\\nodp\\20230203\\LM_aoi13_nodp_0203_meta_lib.pkl'}
            ,

        #sample_dx_fp=r'L:\10_IO\fdsc\outs\ahr_aoi08\post_0124\20230124\ahr_aoi08_post_0124_0124_collect_samples_data.pkl',   
        run_name='post_0203',proj_name='LM_aoi13',
        **kwargs)
    
 
    
    
if __name__=='__main__':
    LM_aoi13_0203()
   
 
    print('done')
