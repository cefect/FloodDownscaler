'''
Created on Dec. 5, 2022

@author: cefect

toy test data
'''
import numpy as np
import pandas as pd
import numpy.ma as ma
from io import StringIO
#from tests.conftest import get_rlay_fp
nan, array = np.nan, np.array

"""setup to construct when called
seems cleaner then building each time we init (some tests dont need these)
more seamless with using real data in tests"""

#===============================================================================
# helpers
#===============================================================================
from hp.tests.tools.rasters import get_mar, get_ar_from_str, get_wse_ar
#===============================================================================
# def get_mar(ar_raw):
#     return ma.array(ar_raw, mask=np.isnan(ar_raw), fill_value=-9999)
# 
# def get_ar_from_str(ar_str, dtype=float):
#     return pd.read_csv(StringIO(ar_str), sep='\s+', header=None).astype(dtype).values
# 
# def get_wse_ar(ar_str, **kwargs):
#     ar1 = get_ar_from_str(ar_str, **kwargs)
#     return np.where(ar1==-9999, np.nan, ar1) #replace nans
#===============================================================================


 
    


#===============================================================================
# raw data
#===============================================================================
dem1_ar = get_mar(
    get_ar_from_str("""
    1    1    1    9    9    9    
    1    1    1    9    9    9
    1    1    1    2    2    9
    2    2    2    9    2    9
    6    2    2    9    2    9
    2    2    2    9    2    9
    4    4    4    2    2    9
    4    4    4    9    9    9
    4    4    4    9    9    1
    """))


wse2_ar = get_mar( #get_wse_ar converts 9999 to null. get_mar converts back to -9999
    get_wse_ar("""
    3    -9999
    4    -9999
    5    -9999    
    """))

"""dummy validation against wse1_ar3
1FP, 1FN"""
wse1_arV = get_mar(
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



#===============================================================================
# intermittent data
#===============================================================================
"""p1_downscale_wetPartials"""
wse1_ar2 =get_mar(
    np.array([
        [ 3.,  3.,  3., np.nan, np.nan, np.nan],
       [ 3.,  3.,  3., np.nan, np.nan, np.nan],
       [ 3.,  3.,  3., np.nan, np.nan, np.nan],
       [ 4.,  4.,  4., np.nan, np.nan, np.nan],
       [nan,  4.,  4., np.nan, np.nan, np.nan],
       [ 4.,  4.,  4., np.nan, np.nan, np.nan],
       [ 5.,  5.,  5., np.nan, np.nan, np.nan],
       [ 5.,  5.,  5., np.nan, np.nan, np.nan],
       [ 5.,  5.,  5., np.nan, np.nan, np.nan]])
    )

"""phase2: _null_dem_violators"""
wse1_ar3 = get_mar(
    array([
        [ 3.,  3.,  3., nan, nan, nan],
       [ 3.,  3.,  3., nan, nan, nan],
       [ 3.,  3.,  3.,  3.,  3., nan],
       [ 4.,  4.,  4., nan,  4., nan],
       [nan,  4.,  4., nan,  4., nan],
       [ 4.,  4.,  4., nan,  4., nan],
       [ 5.,  5.,  5.,  5.,  5., nan],
       [ 5.,  5.,  5., nan, nan, nan],
       [ 5.,  5.,  5., nan, nan,  5.]])
    )





