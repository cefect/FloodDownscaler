'''
Created on Apr. 15, 2022

@author: cefect
'''
import os, shutil
 
import pandas as pd
import numpy as np
from scipy.stats import norm
import qgis.core
#===============================================================================
# setup matplotlib
#===============================================================================
 
import matplotlib
matplotlib.use('Qt5Agg') #sets the backend (case sensitive)
matplotlib.set_loglevel("info") #reduce logging level
import matplotlib.pyplot as plt

#set teh styles
plt.style.use('default')

#font
matplotlib_font = {
        'family' : 'serif',
        'weight' : 'normal',
        'size'   : 8}

matplotlib.rc('font', **matplotlib_font)
matplotlib.rcParams['axes.titlesize'] = 10 #set the figure title size
matplotlib.rcParams['figure.titlesize'] = 16
matplotlib.rcParams['figure.titleweight']='bold'

#spacing parameters
matplotlib.rcParams['figure.autolayout'] = False #use tight layout

#legends
matplotlib.rcParams['legend.title_fontsize'] = 'large'

print('loaded matplotlib %s'%matplotlib.__version__)


 


def ecWrkr(
        n=int(1e3),
           mean=10.0,
           ): #build an ErrorCalcs
    
    
    #build trues
    pd.Series(mean, index=range(n))
    
    #build samples
    rv_norm = norm(loc=mean, scale=1.0)
    
    mean, var, skew, kurt = rv_norm.stats(moments='mvsk')
 
    """
    plt.close()
    fig.show()
    """
    
    x = np.linspace(rv_norm.ppf(0.01),rv_norm.ppf(0.99), 100)
    y = rv_norm.pdf(x)
    
    fig, ax = plt.subplots(1, 1)
    ax.plot(x, y,'r-', lw=5, alpha=0.6, label='norm pdf')
ecWrkr()