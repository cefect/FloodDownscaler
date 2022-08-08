'''
Created on Aug. 7, 2022

@author: cefect
'''


import pytest, tempfile, datetime, os, copy
from hp.rio import RioWrkr
from rasterio.enums import Resampling

test_rlay_fp= r'C:\LS\09_REPOS\01_COMMON\coms\tests\data\scratch.tif'

test_rlay_fp_l=[
    r'C:\LS\09_REPOS\01_COMMON\coms\tests\data\lay2.tif',
    r'C:\LS\09_REPOS\01_COMMON\coms\tests\data\lay1.tif']

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


@pytest.mark.parametrize('rlay_ref_fp', [test_rlay_fp], indirect=True)
@pytest.mark.parametrize('resampling', [
    Resampling.bilinear, Resampling.nearest])
@pytest.mark.parametrize('scale', [2, 10])
def test_upsample(riowrkr, resampling, scale):
    
    ofp = riowrkr.resample(resampling=resampling, scale=scale, write=True)
    
    assert os.path.exists(ofp)
 

@pytest.mark.dev
@pytest.mark.parametrize('rlay_ref_fp', [None], indirect=True)
@pytest.mark.parametrize('test_rlay_fp_l', [test_rlay_fp_l])
 
def test_merge(riowrkr, test_rlay_fp_l):
    
    #remove the first
    #base_fp = test_rlay_fp_l.pop(0)
    
    #load remainers
    ds_l = [riowrkr.open_dataset(fp) for fp in test_rlay_fp_l]
    
    #add base back
    #ds_l.append(riowrkr._base())
    
    
    ofp = riowrkr.merge(ds_l)
    
    assert os.path.exists(ofp)