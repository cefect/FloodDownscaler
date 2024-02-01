'''
Created on Mar. 27, 2023

@author: cefect

refactored so the tests are shareable with fdsc
'''
import pytest, copy, os, random, re
xfail = pytest.mark.xfail

import pandas as pd
from pandas import IndexSlice as idx

from fperf.plot.pipeline import PostSession as Session

from hp.pd import view

from hp.tests.tools.rasters import get_rlay_fp

from definitions import src_dir
if 'L:' in src_dir:
    """issues with relative paths"""
    src_dir = src_dir.replace('L:', 'C:\\LS')
    
#===============================================================================
# test data
#===============================================================================
from fperf.tests.data.toy import dem1_ar, crs_default, bbox_default
gfp = lambda ar, name:get_rlay_fp(ar, name, crs=crs_default, bbox=bbox_default)
toy_d = {'dem':gfp(dem1_ar, 'dem1')}

vali_multi_pick_fp = os.path.join(src_dir, 
          r'fperf\tests\data\test_run_vali_multi\test_testrun_0327_rvX.pkl')
 
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



#===============================================================================
# test_load_run_serx
#===============================================================================
@pytest.fixture(scope='function')
def _load_run_serx(ses):
    def func(pick_fp):
        assert os.path.exists(pick_fp)
        return ses.load_run_serx(pick_fp, base_dir=os.path.dirname(pick_fp))
    return func

@pytest.mark.parametrize('pick_fp', [vali_multi_pick_fp])
def test_load_run_serx(pick_fp, _load_run_serx):
    serx = _load_run_serx(pick_fp)
    
    #===========================================================================
    # serx.to_pickle(
    #     r'l:\09_REPOS\03_TOOLS\FloodGridPerformance\fperf\tests\data\test_run_vali_multi\serx.pkl'
    #     )
    #===========================================================================

serx_pick_fp = os.path.join(src_dir, r'fperf\tests\data\test_run_vali_multi\serx.pkl')

#===============================================================================
# plot_grids_mat
#===============================================================================
@pytest.fixture(scope='function')
def _plot_grids_mat(ses):
    def func(serx_fp, dem_fp):
        serx = pd.read_pickle(serx_fp)
        #filepaths
        gridk = 'wsh'
        fp_d = serx['raw']['fp'].loc[idx[:, gridk]].to_dict()
        return ses.plot_grids_mat(fp_d, gridk=gridk.upper(), dem_fp=dem_fp, 
            inun_fp=serx['raw']['fp'].loc[idx[:, 'inun']].iloc[0])
    return func


                            

@pytest.mark.parametrize('serx_fp, dem_fp', [
    (serx_pick_fp, toy_d['dem'])
    ])
def test_plot_grids_mat(serx_fp,dem_fp, _plot_grids_mat):
    _plot_grids_mat(serx_fp, dem_fp)

#===============================================================================
# plot_inun_perf_mat
#===============================================================================
 

@pytest.fixture(scope='function')
def _plot_inun_perf_mat(ses):
    def __plot_inun_perf_mat(serx_fp, dem_fp):
        serx = pd.read_pickle(serx_fp)
        gridk = 'wsh'
        fp_df, metric_lib = ses.collect_inun_data(serx, gridk, raw_coln='raw')
        fp_df= fp_df.rename(columns={'confusion':'CONFU', 'wsh':'WSH'}) #fix remapping
        return ses.plot_inun_perf_mat(fp_df, metric_lib=metric_lib)
    return __plot_inun_perf_mat


@pytest.mark.parametrize('serx_fp, dem_fp', [
    (serx_pick_fp, toy_d['dem'])
])
def test_plot_inun_perf_mat(serx_fp, dem_fp, _plot_inun_perf_mat):
    _plot_inun_perf_mat(serx_fp, dem_fp)
    
    
#===============================================================================
# plot_HWM_scatter
#===============================================================================
 
@pytest.fixture(scope='function')
def _plot_HWM_scatter(ses):
    def func(serx_fp, dem_fp):
        serx = pd.read_pickle(serx_fp)
        hwm_gdf = ses.collect_HWM_data(
            serx['hwm']['fp'],
            write=False,
        )
        return ses.plot_HWM_scatter(hwm_gdf)
    return func

@pytest.mark.dev
@pytest.mark.parametrize('serx_fp, dem_fp', [
    (serx_pick_fp, toy_d['dem'])
])
def test_plot_HWM_scatter(serx_fp, dem_fp, _plot_HWM_scatter):
    _plot_HWM_scatter(serx_fp, dem_fp)
    
    
    
    
    
    
    