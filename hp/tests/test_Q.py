'''
Created on Oct. 23, 2022

@author: cefect
'''
import pytest, tempfile, datetime, os, copy, logging
from hp.Q import QSession
#===============================================================================
# fixtures
#===============================================================================
@pytest.fixture(scope='session')
def logger(): 
    logger= logging.getLogger('root')
    logger.setLevel(logging.DEBUG) #set the base level for the logger
    logger.info('built test session logger')
    
    return logger


@pytest.mark.dev
 
def test_session(logger, tmp_path,
                  #pytest-qgis fixtures           
            qgis_app, qgis_processing,
            ):
    """need qgis_app to avoid crash"""
    
    with QSession(logger=logger, out_dir=tmp_path, wrk_dir=tmp_path, tmp_dir=tmp_path,
                  qgis_app=qgis_app) as o:
        o._install_info()
        
        #=======================================================================
        # d = o._get_init_pars()
        # assert isinstance(d, dict)
        # assert len(d)>0
        #=======================================================================