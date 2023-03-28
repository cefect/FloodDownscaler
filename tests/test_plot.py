'''
Created on Mar. 27, 2023

@author: cefect

mostly an implementation of fperf.tests.test_plot
'''


import pytest, copy, os, random, re, pickle
import numpy as np
import pandas as pd

xfail = pytest.mark.xfail

from definitions import src_dir
if 'L:' in src_dir:
    """issues with relative paths"""
    src_dir = src_dir.replace('L:', 'C:\\LS')
    
from hp.pd import view
from hp.tests.tools.rasters import get_rlay_fp

from fdsc.plot.control import Fdsc_Plot_Session as Session

#tests from fperf
 

#===============================================================================
# test data
#===============================================================================
#from fperf.tests.data.toy import dem1_ar, crs_default, bbox_default
from fperf.tests.test_plot import dem1_ar, crs_default, bbox_default, toy_d
#===============================================================================
# gfp = lambda ar, name:get_rlay_fp(ar, name, crs=crs_default, bbox=bbox_default)
# toy_d = {'dem':gfp(dem1_ar, 'dem1')}
#===============================================================================



#===============================================================================
# fixtures------------
#===============================================================================

@pytest.fixture(scope='function')
def ses(init_kwargs):    
    """Mock session for tests""" 
    with Session(**init_kwargs, crs=crs_default) as session: 
        yield session
        
#===============================================================================
# TESTS---------
#===============================================================================
def test_init(ses):
    pass

#===============================================================================
# test_load_run_serx
#===============================================================================
#see tests.test_eval.test_run_dsc_vali_multi()
vali_multi_pick_fp = os.path.join(src_dir, 
          r'tests\data\test_run_dsc_vali_multi_toy\test_testrun_0328_gfps.pkl')

#load fixture from file explicitly
from fperf.tests.test_plot import _load_run_serx
pytestmark = pytest.mark.usefixtures("_load_run_serx")

 
@pytest.mark.parametrize('pick_fp', [vali_multi_pick_fp])
def test_load_run_serx(pick_fp, _load_run_serx):
    serx = _load_run_serx(pick_fp)
    
    serx.to_pickle(
        os.path.join(os.path.dirname(vali_multi_pick_fp), 'serx.pkl') 
        )
    
serx_pick_fp = os.path.join(src_dir, r'tests\data\test_run_dsc_vali_multi_toy\serx.pkl')

#===============================================================================
# plot_grids_mat
#===============================================================================
from fperf.tests.test_plot import _plot_grids_mat
pytestmark = pytest.mark.usefixtures("_plot_grids_mat")

 
@pytest.mark.parametrize('serx_fp, dem_fp', [
    (serx_pick_fp, toy_d['dem'])
    ])
def test_plot_grids_mat(serx_fp,dem_fp, _plot_grids_mat):
    _plot_grids_mat(serx_fp, dem_fp)

#===============================================================================
# plot_inun_perf_mat
#===============================================================================
from fperf.tests.test_plot import _plot_inun_perf_mat
pytestmark = pytest.mark.usefixtures("_plot_inun_perf_mat")

 
@pytest.mark.parametrize('serx_fp, dem_fp', [
    (serx_pick_fp, toy_d['dem'])
])
def test_plot_inun_perf_mat(serx_fp, dem_fp, _plot_inun_perf_mat):
    _plot_inun_perf_mat(serx_fp, dem_fp)
    
#===============================================================================
# plot_HWM_scatter
#===============================================================================
from fperf.tests.test_plot import _plot_HWM_scatter
pytestmark = pytest.mark.usefixtures("_plot_HWM_scatter")

@pytest.mark.dev
@pytest.mark.parametrize('serx_fp, dem_fp', [
    (serx_pick_fp, toy_d['dem'])
])
def test_plot_HWM_scatter(serx_fp, dem_fp, _plot_HWM_scatter):
    _plot_HWM_scatter(serx_fp, dem_fp)














