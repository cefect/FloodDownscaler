'''
Created on Dec. 5, 2022

@author: cefect

tests for whitebox
'''


import pytest, copy, os

from fdsc.wbt import WBT_worker

#===============================================================================
# tests-------
#===============================================================================
@pytest.mark.dev
def test_wbt():
    WBT_worker()
    
    


#===============================================================================
# @pytest.mark.parametrize('dem_ar, wse_ar', [
#     (proj_lib['dem1'], proj_lib['wse1'])
#     ])
# def test_runr(dem_ar, wse_ar, tmp_path):
#     pass
#===============================================================================