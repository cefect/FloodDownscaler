'''
Created on Aug. 29, 2022

@author: cefect
'''


import pytest, tempfile, datetime, os, copy, logging
from hp.np import *

#===============================================================================
# @pytest.mark.parametrize('ar', [
#     np.random.random((4*3,4*10))*10
#     ])
#===============================================================================
@pytest.mark.dev
@pytest.mark.parametrize('func', [np.max, np.min])
@pytest.mark.parametrize('n', [2, 3, 10])
@pytest.mark.parametrize('k', [2, 10**2])
@pytest.mark.parametrize('j', [2, 3])
def test_blockwise(func, n, k, j):
    ar = (np.random.random((n*k ,n*j))*10).astype(int)
    apply_blockwise(ar, func, downscale=n)
    


@pytest.mark.parametrize('n', [2,   10])
@pytest.mark.parametrize('k', [2,   10**2])
@pytest.mark.parametrize('j', [  3, 10**3])
def test_upsample(n, k, j):
    ar = (np.random.random((k ,j))*10).astype(int)
     
    res_ar = downsample(ar,n=n) 
