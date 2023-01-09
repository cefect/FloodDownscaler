'''
Created on Jan. 9, 2023

@author: cefect
'''


import pytest, copy, os, random, re
xfail = pytest.mark.xfail

import numpy as np


from tests.conftest import (
      proj_lib, get_rlay_fp, crs_default, get_aoi_fp
    )

from fdsc.analysis.post import PostSession, run_post



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
    
    with PostSession(
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