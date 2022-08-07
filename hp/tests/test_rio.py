'''
Created on Aug. 7, 2022

@author: cefect
'''


import pytest, tempfile, datetime, os, copy
from hp.rio import RioWrkr

#===============================================================================
# fixtures
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

@pytest.mark.parametrize('rlay_ref_fp', [r'C:\Users\cefect\Downloads\scratch.tif'], indirect=True)
def test_init(riowrkr):
    
    assert isinstance(riowrkr.ref_name, str)


@pytest.mark.parametrize('rlay_ref_fp', [r'C:\Users\cefect\Downloads\scratch.tif'], indirect=True)
def test_upsample(riowrkr):
    
    assert isinstance(riowrkr.ref_name, str)