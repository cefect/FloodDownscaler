'''
Created on Jan. 7, 2023

@author: cefect
'''


import pytest, copy, os, random, re
xfail = pytest.mark.xfail
import shapely.geometry as sgeo


from tests.conftest import (
      proj_lib, get_rlay_fp, crs_default, get_aoi_fp, par_algoMethodKwargs
    )

from fdsc.analysis.pipeline import run_pipeline_multi
from fdsc.analysis.pipeline import PipeSession as Session

from hp.tests.tools.rasters import get_poly_fp_from_rlay
#===============================================================================
# test data
#===============================================================================
td1 = proj_lib['fred01']

from tests.data.toy import dem1_ar, wse2_ar, wse1_arV
dem1_rlay_fp = get_rlay_fp(dem1_ar, 'dem1') 
wse2_rlay_fp = get_rlay_fp(wse2_ar, 'wse2')
wse1_rlayV_fp = get_rlay_fp(wse1_arV, 'wse1V')
toy_aoi_fp = get_aoi_fp(sgeo.box(0, 30, 60, 60))
inun_poly_fp = get_poly_fp_from_rlay(wse1_rlayV_fp)

#===============================================================================
# fixtures------------
#===============================================================================
@pytest.fixture(scope='function')
def wrkr(tmp_path,write,logger, test_name,
         crs= crs_default,
                    ):
    
    """Mock session for tests"""
 
    #np.random.seed(100)
    #random.seed(100)
    
    with Session(
                 #oop.Basic
                 out_dir=tmp_path, 
                 tmp_dir=os.path.join(tmp_path, 'tmp_dir'),
                  proj_name='test', #probably a better way to propagate through this key 
                 run_name=test_name[:8].replace('_',''),                  
                 relative=True, write=write, #avoid writing prep layers                 
                 logger=logger, overwrite=True,
                   
                   #oop.Session
                   logfile_duplicate=False,
                   
                   #RioSession
                   crs=crs, 
                   ) as ses:
 
        yield ses

#===============================================================================
# tests--------
#===============================================================================


@pytest.mark.parametrize('raster_fp_d, aoi_fp', [
    ({'dem1':dem1_rlay_fp, 'wse2':wse2_rlay_fp, 'wse1':wse1_rlayV_fp},toy_aoi_fp)
    ])
def test_clip_set(raster_fp_d, aoi_fp, 
                 wrkr):
    wrkr.clip_set(raster_fp_d, aoi_fp=aoi_fp)


 
@pytest.mark.parametrize('dem1_fp, wse2_fp, true_wse_fp, true_inun_fp, sample_pts_fp, aoi_fp', [

    (dem1_rlay_fp, wse2_rlay_fp, wse1_rlayV_fp, inun_poly_fp, None,  None),
    (td1['dem1_rlay_fp'], td1['wse2_rlay_fp'], td1['wse1_rlayV_fp'], td1['inun_vlay_fp'], td1['sample_pts_fp'], td1['aoi_fp'])
 
    ])
@pytest.mark.parametrize(*par_algoMethodKwargs)
def test_run_dsc_vali(dem1_fp, wse2_fp, true_wse_fp, true_inun_fp, sample_pts_fp, aoi_fp,
              method, kwargs, #from par_algoMethodKwargs
              wrkr):    
    wrkr.run_dsc_vali(wse2_fp, dem1_fp, 
                  
                 aoi_fp=aoi_fp, 
 
                 dsc_kwargs=dict(method = method, rkwargs = kwargs),
                 vali_kwargs=dict(true_wse_fp=true_wse_fp, true_inun_fp=true_inun_fp, sample_pts_fp=sample_pts_fp),
                 )

@pytest.mark.dev
@pytest.mark.parametrize('dem1_fp, wse_fp,  true_inun_fp, sample_pts_fp, aoi_fp', [

    #(dem1_rlay_fp, wse2_rlay_fp, wse1_rlayV_fp, inun_poly_fp, None,  None),
    (td1['dem1_rlay_fp'], td1['wse2_rlay_fp'], td1['inun_vlay_fp'], None, td1['aoi_fp']),
    (td1['dem1_rlay_fp'], td1['wse1_rlayV_fp'],  td1['inun_vlay_fp'],None, td1['aoi_fp'])
 
    ])
def test_run_hyd_vali(dem1_fp, wse_fp,  true_inun_fp, sample_pts_fp, aoi_fp, 
              wrkr):    
    
    wrkr.run_hyd_vali(wse_fp, dem1_fp, 
                  
                 aoi_fp=aoi_fp, 
 
                 vali_kwargs=dict(true_inun_fp=true_inun_fp, sample_pts_fp=sample_pts_fp),
                 )
    
@pytest.mark.parametrize('dem1_fp, wse2_fp, true_wse_fp, true_inun_fp, sample_pts_fp, aoi_fp', [

    (dem1_rlay_fp, wse2_rlay_fp, wse1_rlayV_fp, inun_poly_fp, None,  None),
    (td1['dem1_rlay_fp'], td1['wse2_rlay_fp'], td1['wse1_rlayV_fp'], td1['inun_vlay_fp'], td1['sample_pts_fp'], td1['aoi_fp'])
 
    ])

def test_runr_multi(dem1_fp, wse2_fp, true_wse_fp, true_inun_fp, sample_pts_fp, aoi_fp,
                    tmp_path, logger):
    
    method_pars = {e[0]:e[1] for e in par_algoMethodKwargs[1]}
    
        
    run_pipeline_multi(wse2_fp, dem1_fp, 
                  
                 aoi_fp=aoi_fp, 
                 method_pars=method_pars,
                 vali_kwargs=dict(true_wse_fp=true_wse_fp, true_inun_fp=true_inun_fp, sample_pts_fp=sample_pts_fp),
                 out_dir=tmp_path,tmp_dir=os.path.join(tmp_path, 'tmp_dir'),logger=logger,logfile_duplicate=False,
                 )
    