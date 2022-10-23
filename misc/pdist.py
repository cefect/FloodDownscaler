'''
Created on Aug. 19, 2022

@author: cefect
'''

import numpy as np

from scipy.stats import norm

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
matplotlib.rc('font', **{
        'family' : 'serif',
        'weight' : 'normal',
        'size'   : 8})
 
for k,v in {
    'axes.titlesize':10,
    'axes.labelsize':10,
    'figure.titlesize':12,
    'figure.autolayout':False,
    'figure.figsize':(10,6),
    'legend.title_fontsize':'large'
    }.items():
        matplotlib.rcParams[k] = v
  
print('loaded matplotlib %s'%matplotlib.__version__)


 
fig, ax = plt.subplots(1, 1)

mean, var, skew, kurt = norm.stats(moments='mvsk')


x = np.linspace(norm.ppf(0.01),
                norm.ppf(0.99), 100)

ax.plot(x, norm.pdf(x),
       'r-', lw=5, alpha=0.6, label='norm pdf')


 

"""
plt.show()
"""