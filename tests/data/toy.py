'''
Created on Dec. 5, 2022

@author: cefect

toy test data
'''
import numpy as np
import pandas as pd
from io import StringIO
#from tests.conftest import get_rlay_fp

"""setup to construct when called
seems cleaner then building each time we init (some tests dont need these)
more seamless with using real data in tests"""

#===============================================================================
# helpers
#===============================================================================
def get_ar_from_str(ar_str, dtype=float):
    return pd.read_csv(StringIO(ar_str), sep='\s+', header=None).astype(dtype).values

def get_wse_ar(ar_str, **kwargs):
    ar1 = get_ar_from_str(ar_str, **kwargs)
    return np.where(ar1==-9999, np.nan, ar1) #replace nans


 
    


#===============================================================================
# raw data
#===============================================================================
dem1_ar = get_ar_from_str("""
    1    1    1    9    9    9
    1    1    1    9    9    9
    1    1    1    2    2    9
    2    2    2    9    2    9
    6    2    2    9    2    9
    2    2    2    9    2    9
    4    4    4    2    2    9
    4    4    4    9    9    9
    4    4    4    9    9    9
    """)
wse2_ar = get_wse_ar("""
    3    -9999
    4    -9999
    5    -9999    
    """)




#===============================================================================
# intermittent data
#===============================================================================

wse1_ar =np.array([[ 3.,  3.,  3., np.nan, np.nan, np.nan],
       [ 3.,  3.,  3., np.nan, np.nan, np.nan],
       [ 3.,  3.,  3., np.nan, np.nan, np.nan],
       [ 4.,  4.,  4., np.nan, np.nan, np.nan],
       [np.nan,  4.,  4., np.nan, np.nan, np.nan],
       [ 4.,  4.,  4., np.nan, np.nan, np.nan],
       [ 5.,  5.,  5., np.nan, np.nan, np.nan],
       [ 5.,  5.,  5., np.nan, np.nan, np.nan],
       [ 5.,  5.,  5., np.nan, np.nan, np.nan]])