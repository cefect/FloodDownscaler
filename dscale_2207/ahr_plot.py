'''
Created on Mar. 28, 2023

@author: cefect

main runner for plotting the ahr sim results
'''

#===============================================================================
# PLOT ENV------
#===============================================================================

#===============================================================================
# setup matplotlib----------
#===============================================================================
env_type = 'present'
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
        'figure.figsize':(17.7*cm,18*cm),#typical full-page textsize for Copernicus (with 4cm for caption)
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
 
    font_size=14
 
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
from dscale_2207.pipeline import run_plot, load_pick, run_plot_multires, run_plot_inun_aoi
from dscale_2207.ahr import init_kwargs, pick_lib, proj_lib, debug
if debug:print('DEBUGGING!!!!\n\n\n')
zoom_aoi = proj_lib['aoiZ_fp']
#===============================================================================
# vars
#===============================================================================
init_kwargs['run_name'] = init_kwargs['run_name']+'p'
del init_kwargs['aoi_fp']



#===============================================================================
# run
#===============================================================================
def plot_lvl1():    
    """main plots for downscaling from 32m to 4m
    run plotting pipeline
        plot_HWM_scatter
        plot_grids_mat_fdsc
        plot_inun_perf_mat
    """
    
    
    
    #load the eval results pickle
    dsc_vali_res_lib = load_pick(pick_lib['2eval'])
    
 
    #run original (full aoi) pipeline
    res_d1= run_plot(dsc_vali_res_lib, init_kwargs = {**init_kwargs, **env_kwargs},
               
             hwm_scat_kg=dict(
                 style_default_d=dict(marker='o', fillstyle='none', alpha=0.8, color='black'),
                 fig_mat_kwargs=dict(ncols=3),
                 ),
               
             grids_mat_kg=dict(
                 aoi_fp=zoom_aoi,
                 fig_mat_kwargs=dict(ncols=3),
                 vmin=87.5, vmax=91.0,
                 ),
               
             inun_per_kg=dict(
                 box_fp=zoom_aoi,
                fig_mat_kwargs=dict(
                    #=figsize=(25*cm, 25*cm),
                  )),
             )
    
     
    
    #run with clipped aoi
    #===========================================================================
    # run_plot_inun_aoi(
    #     load_pick(pick_lib['3evalF']),
    #     init_kwargs = {**init_kwargs, **env_kwargs},
    #     inun_per_kg=dict(),
    #     )
    #===========================================================================
        
                      
    
    
    
def plot_multiRes():
    """plotting the multi-resolution stats"""
    from dscale_2207.ahr_multiRes import pick_lib
    
    #load the stats results pickle
    dsc_res_lib = load_pick(pick_lib['4stats'])
    
    return run_plot_multires(dsc_res_lib, 
                             init_kwargs = {**init_kwargs, **env_kwargs},
                             )
    
    
    
if __name__=='__main__':
    plot_lvl1()
    
    #plot_multiRes()
    
    
    print('finished ')
    
    
    
    