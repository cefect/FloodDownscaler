'''
Created on Mar. 28, 2023

@author: cefect

plotting for the ahr sim
'''

#===============================================================================
# PLOT ENV------
#===============================================================================

#===============================================================================
# setup matplotlib----------
#===============================================================================
env_type = 'draft'
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
        'figure.figsize':(19*cm,19*cm),#typical full-page textsize for AGU (with 4cm for caption)
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
# imports
#===============================================================================
from dscale_2207.pipeline import run_plot, load_pick
from dscale_2207.ahr import init_kwargs, pick_lib

#===============================================================================
# vars
#===============================================================================
init_kwargs['run_name'] = init_kwargs['run_name']+'p'
del init_kwargs['aoi_fp']

zoom_aoi = r'l:\10_IO\2207_dscale\ins\ahr\aoi13\aoi09t_zoom0308_4647.geojson'

#===============================================================================
# run
#===============================================================================
if __name__=='__main__':
    #load the eval results pickle
    dsc_vali_res_lib = load_pick(pick_lib['2eval'])
    
 
    #run
    result = run_plot(dsc_vali_res_lib, init_kwargs = {**init_kwargs, **env_kwargs},
             
             hwm_scat_kg=dict(
                 style_default_d=dict(marker='o', fillstyle='none', alpha=0.8, color='black'),
                 fig_mat_kwargs=dict(ncols=3),
                 ),
             
             grids_mat_kg=dict(
                 aoi_fp=zoom_aoi,
                 fig_mat_kwargs=dict(ncols=3),
                 ),
             
             inun_per_kg=dict(
                 box_fp=zoom_aoi,
                fig_mat_kwargs=dict(
                    #=figsize=(25*cm, 25*cm),
                  )),
             )
    
    print('finished ')
    
    
    
    
    
    
    