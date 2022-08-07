'''
Created on Aug. 7, 2022

@author: cefect
'''


import pytest, tempfile, datetime, os, copy
from hp.rio import RioWrkr
from rasterio.enums import Resampling

test_rlay_fp= r'C:\LS\09_REPOS\01_COMMON\coms\tests\data\scratch.tif'

#===============================================================================
# fixtures------
#===============================================================================
@pytest.fixture(scope='function')
def riowrkr(session, rlay_ref_fp):
    with RioWrkr(
        rlay_ref_fp=rlay_ref_fp,
        
         #oop.Basic
         session=session
        
        ) as wrkr:
        yield wrkr
        
@pytest.fixture(scope='function')  
def rlay_ref_fp(request):
    return request.param



#===============================================================================
# tests---------
#===============================================================================
@pytest.mark.parametrize('rlay_ref_fp', [test_rlay_fp], indirect=True)
def test_init(riowrkr):
    
    assert isinstance(riowrkr.ref_name, str)

@pytest.mark.dev
@pytest.mark.parametrize('rlay_ref_fp', [test_rlay_fp], indirect=True)
@pytest.mark.parametrize('resampling', [
    Resampling.bilinear, Resampling.nearest])
@pytest.mark.parametrize('scale', [2, 10])
def test_upsample(riowrkr, resampling, scale):
    
    ofp = riowrkr.resample(resampling=resampling, scale=scale, write=True)
    
    assert os.path.exists(ofp)
 