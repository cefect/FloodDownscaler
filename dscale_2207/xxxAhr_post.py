'''
Created on Jan. 9, 2023

@author: cefect

data analysis for Ahr results
'''

import os
import numpy as np
from fdsc.analysis.post import basic_post_pipeline

#===============================================================================
# setup matplotlib----------
#===============================================================================
env_type = 'journal'
cm = 1 / 2.54

if env_type == 'journal': 
    usetex = True
elif env_type == 'draft':
    usetex = False
elif env_type == 'present':
    usetex = False
else:
    raise KeyError(env_type)

 
 
  
import matplotlib
#matplotlib.use('Qt5Agg') #sets the backend (case sensitive)
matplotlib.set_loglevel("info") #reduce logging level
import matplotlib.pyplot as plt
 
#set teh styles
plt.style.use('default')

def set_doc_style():
 
    font_size=8
    matplotlib.rc('font', **{'family' : 'serif','weight' : 'normal','size'   : font_size})
     
    for k,v in {
        'axes.titlesize':font_size,
        'axes.labelsize':font_size,
        'xtick.labelsize':font_size,
        'ytick.labelsize':font_size,
        'figure.titlesize':font_size+2,
        'figure.autolayout':False,
        'figure.figsize':(17*cm,19*cm),#typical textsize for AGU
        'legend.title_fontsize':'large',
        'text.usetex':usetex,
        }.items():
            matplotlib.rcParams[k] = v

#===============================================================================
# journal style
#===============================================================================
if env_type=='journal':
    set_doc_style()
 
 
    env_kwargs=dict(
        output_format='pdf',add_stamp=False,add_subfigLabel=True,transparent=True
        )            
#===============================================================================
# draft
#===============================================================================
elif env_type=='draft':
    set_doc_style() 
 
    env_kwargs=dict(
        output_format='svg',add_stamp=True,add_subfigLabel=True,transparent=True
        )          
#===============================================================================
# presentation style    
#===============================================================================
elif env_type=='present': 
 
    env_kwargs=dict(
        output_format='svg',add_stamp=True,add_subfigLabel=False,transparent=False
        )   
 
    font_size=12
 
    matplotlib.rc('font', **{'family' : 'sans-serif','sans-serif':'Tahoma','weight' : 'normal','size':font_size})
     
     
    for k,v in {
        'axes.titlesize':font_size+2,
        'axes.labelsize':font_size+2,
        'xtick.labelsize':font_size,
        'ytick.labelsize':font_size,
        'figure.titlesize':font_size+4,
        'figure.autolayout':False,
        'figure.figsize':(34*cm,19*cm), #GFZ template slide size
        'legend.title_fontsize':'large',
        'text.usetex':usetex,
        }.items():
            matplotlib.rcParams[k] = v
  
print('loaded matplotlib %s'%matplotlib.__version__)
 
 
#===============================================================================
# funcs----------
#===============================================================================
def runr(meta_fp_d, inun_perf_kwargs=dict(), ses_init_kwargs=dict(), **kwargs):
    """environment defaults for runner""" 
        
    return basic_post_pipeline(meta_fp_d, 
                               inun_perf_kwargs=inun_perf_kwargs,
                               ses_init_kwargs={**ses_init_kwargs, **env_kwargs},                               
                               **kwargs)
    

 

#results data (for present and non-present functions)
 

ahr_aoi08_r32_0303_d =    {'CostGrow': 'L:\\10_IO\\fdsc\\outs\\ahr_aoi08_0303\\cgs\\20230304\\ahr_aoi08_0303_cgs_0304_meta_lib.pkl',
 'Basic': 'L:\\10_IO\\fdsc\\outs\\ahr_aoi08_0303\\rsmp\\20230304\\ahr_aoi08_0303_rsmp_0304_meta_lib.pkl',
 'SimpleFilter': 'L:\\10_IO\\fdsc\\outs\\ahr_aoi08_0303\\rsmpF\\20230304\\ahr_aoi08_0303_rsmpF_0304_meta_lib.pkl',
 'Schumann14': 'L:\\10_IO\\fdsc\\outs\\ahr_aoi08_0303\\s14\\20230304\\ahr_aoi08_0303_s14_0304_meta_lib.pkl',
 'WSE2': 'L:\\10_IO\\fdsc\\outs\\ahr_aoi08_0303\\wse2_vali\\20230304\\ahr_aoi08_0303_wse2_vali_0304_meta_lib.pkl',
 'WSE1': 'L:\\10_IO\\fdsc\\outs\\ahr_aoi08_0303\\wse1_vali\\20230304\\ahr_aoi08_0303_wse1_vali_0304_meta_lib.pkl'}
    
ahr_aoi_fp = r'L:\02_WORK\NRC\2207_dscale\04_CALC\ahr\aoi\aoi09t_zoom0308.geojson'
present_mod_keys = ['Basic', 'Schumann14', 'CostGrow', 'WSE1']
present_rowLabels_d = {'WSE1':'Hydrodyn. (s1)', 'Basic':'Hydrodyn. (s2)'}
 

def ahr_aoi08_0303_present(**kwargs):
    return runr(ahr_aoi08_r32_0303_d,
 
        ses_init_kwargs = dict(
            run_name='present',proj_name='ahr_aoi08_0303',
            ),
        
        hwm3_kwargs=dict(
            mod_keys = present_mod_keys,rowLabels_d = present_rowLabels_d,
            fig_mat_kwargs=dict(ncols=2,total_fig_width=19),
            metaKeys_l=['rmse'],
            ),
        
        inun_perf_kwargs=dict(
            row_keys=present_mod_keys,rowLabels_d = present_rowLabels_d,
            
            add_subfigLabel=False,  figsize=(20*cm,18*cm),
            pie_legend=False,arrow1=False,
            ),
        
        rlay_res_kwargs=dict(
             aoi_fp=ahr_aoi_fp,output_format='png',
            #mod_keys = present_mod_keys,
            fig_mat_kwargs=dict(figsize=(25*cm,19*cm), add_subfigLabel=False),
            ),
 
        **kwargs)
    

def ahr_aoi08_0303(**kwargs):
 
    return runr(ahr_aoi08_r32_0303_d, 
        ses_init_kwargs=dict(run_name='post_0303',proj_name='ahr_aoi08_0303'),
        rlay_res_kwargs=dict(aoi_fp=ahr_aoi_fp, 
                             #output_format='png'
                             ),
        inun_perf_kwargs=dict(box_fp=ahr_aoi_fp), 
        **kwargs)

#===============================================================================
# def ahr11_rim0201_r32_0203(**kwargs):
#     return runr(
#             {
#                 'cgs': 'L:\\10_IO\\fdsc\\outs\\ahr11_0203\\cgs\\20230204\\ahr11_0203_cgs_0204_meta_lib.pkl',
#                  'none': 'L:\\10_IO\\fdsc\\outs\\ahr11_0203\\none\\20230204\\ahr11_0203_none_0204_meta_lib.pkl',
#                  'nodp': 'L:\\10_IO\\fdsc\\outs\\ahr11_0203\\nodp\\20230204\\ahr11_0203_nodp_0204_meta_lib.pkl',
#                  's14':r'l:\10_IO\fdsc\outs\ahr11_0203\s14\20230204\ahr11_0203_s14_0204_meta_lib.pkl',
#                  },
#         #sample_dx_fp=r'L:\10_IO\fdsc\outs\ahr_aoi08_0130\p0130\20230130\ahr_aoi08_0130_p0130_0130_collect_samples_data.pkl',   
#         run_name='p0204',proj_name='ahr11_0203',
#         **kwargs)
#===============================================================================
    
ahr11_rim0206_0304_d = {'CostGrow': 'L:\\10_IO\\fdsc\\outs\\ahr11_rim0206_r32_0304\\cgs\\20230304\\ahr11_rim0206_r32_0304_cgs_0304_meta_lib.pkl',
 'Basic': 'L:\\10_IO\\fdsc\\outs\\ahr11_rim0206_r32_0304\\rsmp\\20230304\\ahr11_rim0206_r32_0304_rsmp_0304_meta_lib.pkl',
 'SimpleFilter': 'L:\\10_IO\\fdsc\\outs\\ahr11_rim0206_r32_0304\\rsmpF\\20230304\\ahr11_rim0206_r32_0304_rsmpF_0304_meta_lib.pkl',
 'Schumann14': 'L:\\10_IO\\fdsc\\outs\\ahr11_rim0206_r32_0304\\s14\\20230304\\ahr11_rim0206_r32_0304_s14_0304_meta_lib.pkl',
 'WSE2': 'L:\\10_IO\\fdsc\\outs\\ahr11_rim0206_r32_0304\\wse2_vali\\20230304\\ahr11_rim0206_r32_0304_wse2_vali_0304_meta_lib.pkl',
 'WSE1': 'L:\\10_IO\\fdsc\\outs\\ahr11_rim0206_r32_0304\\wse1_vali\\20230304\\ahr11_rim0206_r32_0304_wse1_vali_0304_meta_lib.pkl'}

def ahr11_rim0206_0304(**kwargs):
    return runr(ahr11_rim0206_0304_d, 
        run_name='post_0304',proj_name='ahr11_rim0206_0304',
        **kwargs)
    
if __name__=='__main__':
 
    
    ahr_aoi08_0303()
    #ahr_aoi08_0303_present()
    
    #ahr11_rim0206_0304()
   
 
    print('done')