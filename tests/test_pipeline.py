'''
Created on Jan. 7, 2023

@author: cefect
'''


import pytest, copy, os, random, re
xfail = pytest.mark.xfail

from tests.conftest import (
      proj_lib, get_rlay_fp 
    )

from fdsc.scripts.pipeline import run_dsc_vali
#===============================================================================
# test data
#===============================================================================
from tests.data.toy import dem1_ar, wse2_ar, wse1_arV
dem1_rlay_fp = get_rlay_fp(dem1_ar, 'dem1') 
wse2_rlay_fp = get_rlay_fp(wse2_ar, 'wse2')
wse1_rlayV_fp = get_rlay_fp(wse1_arV, 'wse1V')


#===============================================================================
# tests--------
#===============================================================================

@pytest.mark.dev
@pytest.mark.parametrize('dem1_fp, wse2_fp, wse1V_fp', [
    (dem1_rlay_fp, wse2_rlay_fp, wse1_rlayV_fp),
    (proj_lib['fred01']['dem1_rlay_fp'], proj_lib['fred01']['wse2_rlay_fp'], proj_lib['fred01']['wse1_rlayV_fp'])
    ])
@pytest.mark.parametrize('dryPartial_method', [
    'costDistanceSimple','none'
    ])
def test_runr(dem1_fp, wse2_fp, wse1V_fp, dryPartial_method,
              tmp_path):    
    run_dsc_vali(wse2_fp, dem1_fp, wse1V_fp = wse1V_fp, 
                 out_dir=tmp_path, run_name='test',
                 dsc_kwargs=dict(dryPartial_method = dryPartial_method),
                 )