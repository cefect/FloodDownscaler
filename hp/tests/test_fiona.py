'''
Created on Apr. 29, 2023

@author: cefect
'''


import pytest, tempfile, datetime, os, copy, math

from hp.basic import dstr
from hp.fiona import SpatialBBOXWrkr, get_bbox_and_crs, write_bbox_vlay
from hp.oop import Session


#===============================================================================
# test data
#===============================================================================
from hp.tests.data.toy_rasters import aoi_box

#===============================================================================
# helpers-----
#===============================================================================

class FionaTestSession(SpatialBBOXWrkr, Session):
    pass

#===============================================================================
# fixtures------
#===============================================================================
@pytest.fixture(scope='function')
def ses(init_kwargs, aoi_fp):
    with FionaTestSession(aoi_fp=aoi_fp,**init_kwargs) as ses:
        yield ses
        
        
@pytest.fixture(scope='session')
def aoi_fp(tmpdir_factory, crs):
    return write_bbox_vlay(aoi_box, crs, os.path.join(tmpdir_factory.mktemp('fiona'), 'aoi')) 
        
#===============================================================================
# tests---------
#===============================================================================
@pytest.mark.dev
def test_init(ses):
    ses.assert_valid_atts()    
    print(dstr(ses.bbox))