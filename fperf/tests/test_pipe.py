'''
Created on Jan. 6, 2023

@author: cefect
'''

import pytest, copy, os, random, re
xfail = pytest.mark.xfail
 
 
from hp.hyd import get_wsh_rlay, write_inun_rlay, write_rlay_to_poly
from hp.tests.tools.rasters import get_rlay_fp
from hp.tests.tools.vectors import get_aoi_fp


from fperf.pipeline import ValidateSession, run_validator
 
from fperf.tests.conftest import proj_lib, temp_dir, get_hwm_random_fp

#===============================================================================
# test data-------
#===============================================================================
 
 
#===============================================================================
# fred data
#===============================================================================
td1 = proj_lib['fred01'].copy()
 
#convert  Fred WSE to depths
f = lambda wse_fp:get_wsh_rlay(td1['dem1_rlay_fp'], wse_fp)
td1_wd1_rlayV_fp = f(td1['wse1_rlayV_fp'])
td1_wd1_rlay3_fp = f(td1['wse1_rlay3_fp'])

#to inuns
f = lambda fp, name:write_inun_rlay(fp, 'WSE', ofp=os.path.join(temp_dir, name+'.tif'))
for k,v in {'inun1V':'wse1_rlayV_fp', 'inun13':'wse1_rlay3_fp'}.items():
    td1[k] = f(td1[v], k)


td1_wd_fps = (td1_wd1_rlayV_fp, td1_wd1_rlay3_fp, td1['sample_pts_fp'])
td1_inun1_fps=(td1['inun1V'], td1['inun13'])
#td1_inun2_fps=(td1['inun_vlay_fp'], td1['inun13'])
#===============================================================================
# toy data
#===============================================================================
 
from fperf.tests.data.toy import (
    wse1_arV, wse1_ar3, wse2_ar, dem1_ar, dem2_ar, crs_default, bbox_default, aoi_box
    )

toy_d = dict()

#get rasters
gfp = lambda ar, name:get_rlay_fp(ar, name, crs=crs_default, bbox=bbox_default)

wse1_rlay3_fp = gfp(wse1_ar3, 'wse13')
wse1_rlayV_fp = gfp(wse1_arV, 'wse1V')
wse2_rlay_fp = gfp(wse2_ar, 'wse2')
dem1_rlay_fp = gfp(dem1_ar, 'dem1')
dem2_rlay_fp = gfp(dem2_ar, 'dem2')
#inun_poly_fp = get_poly_fp_from_rlay(wse1_rlayV_fp)

#convert to depths
f = lambda wse_fp:get_wsh_rlay(dem1_rlay_fp, wse_fp)

toy_wd1_rlay3_fp = f(wse1_rlay3_fp)
toy_wd1_rlayV_fp = f(wse1_rlayV_fp)
toy_wd2_rlay_fp = get_wsh_rlay(dem2_rlay_fp, wse2_rlay_fp)

#to INUN_RLAY
f = lambda fp, name:write_inun_rlay(fp, 'WSE', ofp=os.path.join(temp_dir, name+'.tif'))
for k, fp in {'inun1V':wse1_rlayV_fp, 'inun13':wse1_rlay3_fp}.items():
    toy_d[k] = f(fp, k)
    
#to INUN_POLY
toy_d['inunP'] = write_rlay_to_poly(wse1_rlayV_fp, 'WSE', ofp=os.path.join(temp_dir, 'inunP.geojson'))

#create hwms
toy_hwm_fp = get_hwm_random_fp(count=5, crs=crs_default, bbox=bbox_default)

#AOI
toy_aoi_fp = get_aoi_fp(aoi_box, crs=crs_default)

#package
toy_wd_fps = (toy_wd1_rlayV_fp, toy_wd1_rlay3_fp, None)
toy_inun1_fps = (toy_d['inun1V'], toy_d['inun13'])
#toy_inun2_fps = (inun_poly_fp, wse1_rlay3_fp)

#===============================================================================
# fixtures------------
#===============================================================================

@pytest.fixture(scope='function')
def ses(init_kwargs):    
    """Mock session for tests""" 
    with ValidateSession(**init_kwargs) as session: 
        yield session
 
 
    
#===============================================================================
# test.pipeline----
#===============================================================================
 
@pytest.mark.parametrize('true_wd_fp, pred_wd_fp, sample_pts_fp', [
    td1_wd_fps,
    #toy_wd_fps, #no points
    ]) 
def test_run_vali_pts(true_wd_fp, pred_wd_fp, sample_pts_fp, ses):
    ses.run_vali_pts(sample_pts_fp, true_wd_fp=true_wd_fp, pred_wd_fp=pred_wd_fp)



@pytest.mark.parametrize('pred_wd_fp, hwm_pts_fp', [
    (td1_wd1_rlay3_fp, td1['hwm_pts_fp'])
    ]) 
def test_run_vali_hwm(pred_wd_fp, hwm_pts_fp, ses):
    ses.run_vali_hwm(pred_wd_fp, hwm_pts_fp)
    
@pytest.mark.dev
@pytest.mark.parametrize('true_inun_fp, pred_inun_fp', [
    td1_inun1_fps,
    #td1_inun2_fps,
    toy_inun1_fps,
    #toy_inun2_fps
    ]) 
def test_run_vali_inun(true_inun_fp, pred_inun_fp, ses):
    ses.run_vali_inun(true_inun_fp=true_inun_fp, pred_inun_fp=pred_inun_fp)
    

#===============================================================================
# """needs to be fixed
# @pytest.mark.parametrize('pred_wse_fp, true_wse_fp, true_inun_fp, sample_pts_fp, dem_fp, hwm_pts_fp', [    
#     (td1['wse1_rlay3_fp'], td1['wse1_rlayV_fp'], td1['inun1V'], td1['sample_pts_fp'], td1['dem1_rlay_fp'], td1['hwm_pts_fp']),
#      (td1['wse1_rlay3_fp'], td1['wse1_rlayV_fp'], None, td1['sample_pts_fp'], td1['dem1_rlay_fp'], td1['hwm_pts_fp']), 
#      (wse1_rlay3_fp, wse1_rlayV_fp, None, None, dem1_rlay_fp, None),   
#  
#     (wse1_rlay3_fp, wse1_rlayV_fp, toy_d['inun1V'], None, dem1_rlay_fp, None),
#     ]) 
# def test_run_vali(pred_wse_fp, true_wse_fp, true_inun_fp, sample_pts_fp, dem_fp,hwm_pts_fp,
#                    ses):
#     ses.run_vali(pred_wse_fp=pred_wse_fp, true_wse_fp=true_wse_fp,
#                  true_inun_fp=true_inun_fp, sample_pts_fp=sample_pts_fp, dem_fp=dem_fp, hwm_pts_fp=hwm_pts_fp)
#===============================================================================

 

@pytest.mark.parametrize('d, hwm_pts_fp, inun_fp, aoi_fp', [
    ({'toy1':toy_wd1_rlay3_fp,'toy2':toy_wd2_rlay_fp}, toy_hwm_fp, toy_d['inunP'], None),
    #({'toy1':toy_wd1_rlay3_fp,'toy2':toy_wd2_rlay_fp}, toy_hwm_fp, toy_d['inunP'], toy_aoi_fp),
    ]) 
def test_run_vali_multi(d, hwm_pts_fp, inun_fp, aoi_fp,
                   ses):
    assert os.path.exists(inun_fp), inun_fp
    ses.run_vali_multi(pred_wsh_fp_d=d, hwm_pts_fp=hwm_pts_fp, inun_fp=inun_fp, aoi_fp=aoi_fp,
                       copy_inputs=True)
    
