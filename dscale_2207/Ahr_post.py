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
env_type='journal'


cm = 1/2.54
usetex=True
if usetex:
    os.environ['PATH'] += R";C:\Users\cefect\AppData\Local\Programs\MiKTeX\miktex\bin\x64"
    output_format='pdf'
    add_stamp=False
else:
    output_format='svg'
    add_stamp=True
    
  
import matplotlib
#matplotlib.use('Qt5Agg') #sets the backend (case sensitive)
matplotlib.set_loglevel("info") #reduce logging level
import matplotlib.pyplot as plt
 
#set teh styles
plt.style.use('default')


#===============================================================================
# journal style
#===============================================================================
if env_type=='journal':
    
    #font
    font_size=8
    matplotlib.rc('font', **{'family' : 'serif','weight' : 'normal','size'   : font_size})
     
    for k,v in {
        'axes.titlesize':10,
        'axes.labelsize':10,
        'xtick.labelsize':8,
        'ytick.labelsize':8,
        'figure.titlesize':12,
        'figure.autolayout':False,
        'figure.figsize':(17*cm,19*cm),
        'legend.title_fontsize':'large',
        'text.usetex':usetex,
        }.items():
            matplotlib.rcParams[k] = v
        
#===============================================================================
# presentation style    
#===============================================================================
elif env_type=='present':
    output_format='png'
    #font
    font_size=12
    #matplotlib.rc('font', **{'family' : 'serif','weight' : 'normal','size'   : 8})
    matplotlib.rc('font', **{'family' : 'sans-serif','sans-serif':'Tahoma','weight' : 'normal','size':font_size})
     
     
    for k,v in {
        'axes.titlesize':font_size+2,
        'axes.labelsize':font_size+2,
        'xtick.labelsize':font_size,
        'ytick.labelsize':font_size,
        'figure.titlesize':font_size+4,
        'figure.autolayout':False,
        'figure.figsize':(20*cm,14*cm),
        'legend.title_fontsize':'large',
        'text.usetex':usetex,
        }.items():
            matplotlib.rcParams[k] = v
  
print('loaded matplotlib %s'%matplotlib.__version__)
 
 
#===============================================================================
# funcs----------
#===============================================================================
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

#results data (for present and non-present functions)
ahr_aoi08_r32_0130_d = {
            'cgs': 'L:\\10_IO\\fdsc\\outs\\ahr_aoi08_0130\\cgs\\20230205\\ahr_aoi08_0130_cgs_0205_meta_lib.pkl',
             's14': 'L:\\10_IO\\fdsc\\outs\\ahr_aoi08_0130\\s14\\20230205\\ahr_aoi08_0130_s14_0205_meta_lib.pkl',
             'none': 'L:\\10_IO\\fdsc\\outs\\ahr_aoi08_0130\\none\\20230205\\ahr_aoi08_0130_none_0205_meta_lib.pkl',
             'nodp': 'L:\\10_IO\\fdsc\\outs\\ahr_aoi08_0130\\nodp\\20230205\\ahr_aoi08_0130_nodp_0205_meta_lib.pkl',
             }
    
def ahr_aoi08_r32_0130_30(**kwargs):
    return basic_post_pipeline(ahr_aoi08_r32_0130_d,
        sample_dx_fp=r'L:\10_IO\fdsc\outs\ahr_aoi08_0130\p0130\20230205\ahr_aoi08_0130_p0130_0205_collect_samples_data.pkl',
        hwm_fp=r'l:\02_WORK\NRC\2207_dscale\04_CALC\ahr\calibrate\hwms\NWR_ahr11_hwm_20220113b_fix.geojson',
        inun_fp=r'L:\02_WORK\NRC\202110_Ahr\01_GEN\01_INOUT\2022 01 13 - Heiko - 2021 data\Ahr_Überflutungsflächen_HW07-2021_LfU712_2021_07_14.shp',   
        run_name='post_0206',proj_name='ahr_aoi08_0130',output_format=output_format,add_stamp=add_stamp,
        **kwargs)
    

def ahr_aoi08_r32_0130_30_present(**kwargs):
    return basic_post_pipeline(ahr_aoi08_r32_0130_d,
        #sample_dx_fp=r'L:\10_IO\fdsc\outs\ahr_aoi08_0130\p0130\20230205\ahr_aoi08_0130_p0130_0205_collect_samples_data.pkl',   
        run_name='present',proj_name='ahr_aoi08_0130',
        rlay_mat_kwargs=dict(
            #row_keys = ['vali', 'none', 's14','cgs' ],
            #col_keys = ['c2', 'c3'],
            
            #pieplots only
            row_keys = ['s14','cgs' ],
            col_keys = ['c1'],
            
            add_subfigLabel=False, 
            transparent=False, figsize=(8*cm,12*cm)),
        samples_mat_kwargs=dict(
            col_keys = ['raw_hist', 'corr_scatter'],add_subfigLabel=False,transparent=False,
            figsize=(24*cm, 16*cm),
            ),
        **kwargs)

def ahr11_rim0201_r32_0203(**kwargs):
    return basic_post_pipeline(
            {
                'cgs': 'L:\\10_IO\\fdsc\\outs\\ahr11_0203\\cgs\\20230204\\ahr11_0203_cgs_0204_meta_lib.pkl',
                 'none': 'L:\\10_IO\\fdsc\\outs\\ahr11_0203\\none\\20230204\\ahr11_0203_none_0204_meta_lib.pkl',
                 'nodp': 'L:\\10_IO\\fdsc\\outs\\ahr11_0203\\nodp\\20230204\\ahr11_0203_nodp_0204_meta_lib.pkl',
                 's14':r'l:\10_IO\fdsc\outs\ahr11_0203\s14\20230204\ahr11_0203_s14_0204_meta_lib.pkl',
                 },
        #sample_dx_fp=r'L:\10_IO\fdsc\outs\ahr_aoi08_0130\p0130\20230130\ahr_aoi08_0130_p0130_0130_collect_samples_data.pkl',   
        run_name='p0204',proj_name='ahr11_0203',
        **kwargs)
    
if __name__=='__main__':
    #ahr11_rim0201_r32_0203()
    
    ahr_aoi08_r32_0130_30()
   
 
    print('done')
