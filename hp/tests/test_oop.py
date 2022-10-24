'''pytest oop module
'''

import pytest, tempfile, datetime, os, copy, logging

from hp.oop import Basic, Session

temp_dir = os.path.join(tempfile.gettempdir(), datetime.datetime.now().strftime('%M%S'))

#===============================================================================
# fixtures
#===============================================================================
@pytest.fixture(scope='session')
def logger(request):
    build=True
    if hasattr(request, 'param'):
        build= request.param
            
    
    
    if build:
        logger= logging.getLogger('root')
        logger.setLevel(logging.DEBUG) #set the base level for the logger
        logger.info('built test session logger')
        return logger
    else:
        return None

#===============================================================================
# tests------
#===============================================================================

@pytest.mark.parametrize('out_dir, tmp_dir, wrk_dir',[
    pytest.param(None, None, None, marks=pytest.mark.xfail(reason='bad params')), 
    [temp_dir, temp_dir, temp_dir]])
@pytest.mark.parametrize('proj_name, run_name, obj_name, fancy_name',[
    pytest.param(None, None, None, None, marks=pytest.mark.xfail),
    ['proj_name', 'run_name', 'obj_name', 'fancy_name']]) 
@pytest.mark.parametrize('overwrite, relative, write',[
    pytest.param( None, None, None, marks=pytest.mark.xfail),
    [True, True, True]])

def test_basic(
        tmp_path, 
        out_dir, tmp_dir, wrk_dir,
        proj_name, run_name, obj_name, fancy_name,
        overwrite, relative, write,
        logger):
    """should all fail except the last"""
    
    o = Basic(out_dir=out_dir, tmp_dir=tmp_dir, wrk_dir=wrk_dir,
          proj_name=proj_name, run_name=run_name, obj_name=obj_name, fancy_name=fancy_name,
          overwrite=overwrite, relative=relative, write=write, logger=logger)
    
    d = o._get_init_pars()
    
    assert isinstance(d, dict)
    assert d['out_dir']==out_dir
    

@pytest.mark.dev
@pytest.mark.parametrize('out_dir, tmp_dir, wrk_dir',[
    #[None,None, None], 
    [temp_dir, os.path.join(temp_dir, 'tmp'), temp_dir]])
@pytest.mark.parametrize('proj_name, run_name, obj_name, fancy_name',[
    #[None, 'r1', 'Ses', None],
    ['proj_name', 'run_name', 'obj_name', 'fancy_name']]) 
@pytest.mark.parametrize('overwrite, relative, write',[
    #[None, None, None, None],
    [True, True, True]])
@pytest.mark.parametrize('logger', [True, False], indirect=True)
@pytest.mark.parametrize('logfile_duplicate', [True, False])
def test_session(
        tmp_path, 
        out_dir, tmp_dir, wrk_dir,
        proj_name, run_name, obj_name, fancy_name,
        overwrite, relative, write,
        logger,
        logfile_duplicate):
    
    o = Session(out_dir=out_dir, tmp_dir=tmp_dir, wrk_dir=wrk_dir,
          proj_name=proj_name, run_name=run_name, obj_name=obj_name, fancy_name=fancy_name,
           overwrite=overwrite, relative=relative, write=write, logger=logger,
           logfile_duplicate=logfile_duplicate)
    
    d = o._get_init_pars()
    
    assert isinstance(d, dict)
 
    
    
 
@pytest.mark.parametrize('logger', [True, False], indirect=True)
@pytest.mark.parametrize('out_dir, tmp_dir, wrk_dir',[
    [None,None, None], 
    [temp_dir, temp_dir, temp_dir]])
@pytest.mark.parametrize('proj_name, run_name, obj_name, fancy_name',[
    [None, None, None, None],
    ['proj_name', 'run_name', 'obj_name', 'fancy_name']]) 
@pytest.mark.parametrize('overwrite, relative, write',[
    [ None, None, None],
    [True, True, True]])
def test_inherit(
        tmp_path, logger,
                out_dir, tmp_dir, wrk_dir,
        proj_name, run_name, obj_name, fancy_name,
        overwrite, relative, write,
        ):
    #build kwargs
    kwargs={'out_dir':out_dir, 'tmp_dir':tmp_dir, 'wrk_dir':wrk_dir,
            'proj_name':proj_name, 'run_name':run_name, 'obj_name':obj_name,'fancy_name':fancy_name,
             'overwrite':overwrite, 'relative':relative, 'write':write}
    
    for k,v in copy.deepcopy(kwargs).items():
        if v is None:
            del kwargs[k]
        
    
    with Session(wrk_dir=tmp_path, logger=logger) as ses:
        #d = ses.init_pars_d.copy()
        
        #init a second child object
        o=Basic(**{**ses._get_init_pars(), **kwargs})
        
        d = o._get_init_pars()
    
        assert isinstance(d, dict)
 
        
        
        