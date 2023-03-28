'''
Created on Feb. 4, 2023

@author: cefect
'''
import numpy as np
nan, array = np.nan, np.array
from hp.tests.tools.rasters import get_mar, get_ar_from_str
 

wse1_mar = get_mar(
    array([
        [ 3.,  3.,  3., nan, nan, nan],
       [ 3.,  3.,  3., nan, nan, nan],
       [ 3.,  3.,  3.,  3.,  3., nan],
       [ 4.,  4.,  4., nan,  3., nan],
       [nan,  4.,  4., nan,  3., nan],
       [ 4.,  4.,  4., nan,  3., nan],
       [ 5.,  5.,  5.,  5.,  5., nan],
       [ 5.,  5.,  5., nan, nan, 5.],
       [ 5.,  5.,  5., nan, nan, nan]])
    )