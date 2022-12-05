'''
Created on Dec. 4, 2022

@author: cefect
'''


from numba import jit
import numpy as np
import time

#===============================================================================
# @jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
# def disag(ar):
#     """disaggregate/downscale an array hydraulically"""
#     
#     for i, x in enumerate(np.nditer(ar, order='K')):
#         print(f'{i}:{k}'
#                           
#     for i in range(a.shape[0]):   # Numba likes loops
#         trace += np.tanh(a[i, i]) # Numba likes NumPy functions
#     return a + trace              # Numba likes NumPy broadcasting
#===============================================================================