'''
Methods for object-oriented-programming

Notes
---------------
#===============================================================================
# INHERITANCE
#===============================================================================
I've spent far too many weeks of my life strugglig with inheritance
    seems too difficult to balance simplicity, flexibility, and functionality
    
2022-10-23: it's that time of the year where I get confused about this. 
    and realize I haven't been following my own advice.
    now I'm thinking I should get rid of these helper scripts all together.
    
    As for oop inheritance, I see two main use cases:
    single-children:
        typical session where each class is spawned once,
        separating 'children' from 'session' objects is not needed
        each init can do setup and attribute assignment
    
    multiple-children
        we have a single session, that spawns multiple instances of the same class
        can be helpful when we have some complex iterative states (e.g., interacting agents)
        some classes may need to separate setup from attribute assignment 
            i.e. some things setup once (at session level), others each time a worker is spawned
        
        in these cases, need separate 'worker' and 'session' class (which inherits the worker)
        worker should be setup to naievly expect the setup variables from the parent session
            only those iterative attributes should be modified in workers init
        most methods should live on the child worker
        
        session should handle all the setup which happens once
            then init a single child worker
            
        any additional child spawning needed by the computation
            should happen in specialized functions
            where the attributes needed by the worker are built and passed by the caller
            as recalling all these attributes can be onerous,
                Basic has an 'init_pars_d' attribute which can store these (for retrival later)
        
    
2021-10-31: settled on top down control
    force the caller object to first extract any attributes they want to pass down
    then run these through the childs init
    i.e., the child is agnostic to the source of the attribute
    this keeps the majority of scripts simple
        scripts that want to get complicated with inheritance can do this at the caller level
    
    using the 'inher_d'  to store 'object adding the requirement':[attn] the object to use if it wants to spawn children
        see get_inher_atts
'''

import os, sys, datetime, gc, copy, pickle, pprint, logging
import logging.config
#from qgis.core import QgsMapLayer
from hp.dirz import delete_dir
from hp.basic import today_str, dstr
from hp.pd import nested_dict_to_dx, view
 
from definitions import src_name
 

import numpy as np
import pandas as pd


#===============================================================================
# functions------------------------------------------------------------------- 
#===============================================================================


class Basic(object): #simple base class

    def __init__(self, 
                 
                 #directories
                 out_dir        = None,
                 tmp_dir       = None,
                 wrk_dir       = None,
                 base_dir       = None,
                 
                 #names/labels
                 proj_name      = None,  
                 run_name       = None,
                 obj_name       = None,   
                 fancy_name     = None,
                 
                 #inheritancee
                 init_pars      = None,
                 subdir         = False,
                 
                 #controls
                 #prec           = None,
                 overwrite      = None,  
                 relative       = None,  
                 write          = True,
                 
                 logger         = None,                 
                 ):
        """
        Initialize a generic class object.
    
        Provides common methods and parameters for object based programming.
        
        TODO: break this apart into simpler responsibilities
            logger?
            directories?
    
        Parameters
        ----------
        wrk_dir: str, optional
            Base directory of the project.  
        out_dir : str, optional
            Directory used for outputs.    
        tmp_dir: str, optional
            Directory for temporary outputs (i.e., cache).  

        proj_name: str, default src_name definitions
            Project name
        run_name: str, default 'r0'
            Label for a specific run or version.
        obj_name: str, default __class__.__name__
            Name of object or worker
        fancy_name: str, default [proj_name]_[run_name]_[obj_name]_[mmdd]
            Name for output prefix
        logger: logging.RootLogger, optional
            Logging worker.

 
        overwrite: bool, default False
            Default behavior when attempting to overwrite a file
        relative: bool, default False
            Default behavior of filepaths (relative vs. absolute)

        init_pars: list,
             Names of attributes set by init. useful for spawning children
            
        subdir: bool, default False
            whether to create subidrectories (in the session defaults) using obj_name
        
        """
        
        #=======================================================================
        # personal
        #=======================================================================
        if init_pars is None: init_pars=list()
        self.start = datetime.datetime.now()
        self.today_str = today_str
        
        #=======================================================================
        # basic attachments
        #=======================================================================
        #self.session=session
        
 
        def attach(attVal, attName, directory=False, typeCheck=None, subdir=False):
            #get from session
            if attVal is None:
                return
 
                #assert not session is None, 'for \'%s\' passed None but got no session'%attName
                #attVal = getattr(session, attName)
                
            #make sub directories
            if subdir and directory:
                attVal = os.path.join(attVal, obj_name)
            elif subdir:
                attVal = attVal + '_' + obj_name
                
            #attach
            assert not attVal is None, attName
            setattr(self, attName, attVal)
            init_pars.append(attName)
            
            #handle directories
            if directory:
                if not os.path.exists(attVal):
                    os.makedirs(attVal)
            
            #check
            if not typeCheck is None:
                assert isinstance(getattr(self, attName), typeCheck), \
                    'bad type on \'%s\': %s'%(attName, type(getattr(self, attName)))
            
 
        #=======================================================================
        # basic attachance
        #=======================================================================
        attach(logger,  'logger', typeCheck=None)
        attach(overwrite,  'overwrite', typeCheck=bool)
        #attach(prec,       'prec', typeCheck=int)
        attach(relative,   'relative', typeCheck=bool)
        attach(write,      'write', typeCheck=bool)
        attach(proj_name,  'proj_name', typeCheck=str)
        attach(run_name,   'run_name', typeCheck=str)
        attach(fancy_name, 'fancy_name', typeCheck=str, subdir=subdir)
 
        attach(wrk_dir,    'wrk_dir', directory=True)
        attach(out_dir,    'out_dir', directory=True, subdir=subdir)
        attach(tmp_dir,    'tmp_dir', directory=True, subdir=subdir)
        attach(base_dir,    'base_dir', directory=True)
 
        if obj_name is None:
            obj_name = self.__class__.__name__
        self.obj_name=obj_name
            
        #self.logger=logger

        #=======================================================================
        # wrap
        #=======================================================================
            
        #self._install_info()
        self.init_pars=init_pars
 
        if not logger is None:
            self.logger.debug(f'finished Basic.__init__ w/ out_dir={out_dir}\n    %s '%init_pars)
 
        
    def _get_init_pars(self):
        """only for simple atts... no containers"""
        return {k:getattr(self, k) for k in self.init_pars}.copy()
    
    def _install_info(self,
                         log = None): #print version info
        if log is None: log = self.logger
 
        d = {'python':sys.version, 'numpy':np.__version__, 'pandas':pd.__version__,
             'exe':sys.executable}
 
        txt = pprint.PrettyPrinter(indent=4).pformat(d)
        log.info(txt)
        #systenm paths
        for k in sys.path: 
            log.info('    %s'%k)
        
    def _func_setup(self, dkey, 
                    logger=None, out_dir=None, tmp_dir=None,ofp=None, 
                     resname=None,ext='.tif',subdir=False,
                    ):
        """common function default setup
        
        Parameters
        ----------
        subdir: bool, default False
            build the out_dir as a subdir of the default out_dir (using dkey)
            
        
        Use
        ---------
        log, tmp_dir, out_dir, ofp, resname  = self._func_setup('myFuncName', **kwargs) 
            
        TODO
        ----------
        setup so we can inherit additional parameters from parent classes
        
        see example in RioSession._get_defaults
        
 
        
        """
        #=======================================================================
        # #logger
        #=======================================================================
        if logger is None:
            logger = self.logger
        log = logger.getChild(dkey)
        
        #=======================================================================
        # #temporary directory
        #=======================================================================
        if tmp_dir is None:
            tmp_dir = os.path.join(self.tmp_dir, dkey)
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
            
        #=======================================================================
        # out_dir
        #=======================================================================
        if out_dir is None: 
            out_dir = self.out_dir
        
        if subdir:
            out_dir = os.path.join(out_dir, dkey)
            
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
            
        #=======================================================================
        # ofp
        #=======================================================================
        
        if resname is None:
            resname = self._get_resname(dkey=dkey)
         
        if ofp is None:
            ofp = self._get_ofp(dkey=dkey, out_dir=out_dir, resname=resname, ext=ext) 
 
        #=======================================================================
        # if os.path.exists(ofp):
        #     log.warning('ofp exists... overwriting')
        #     os.remove(ofp)
        #=======================================================================
            
        return log, tmp_dir, out_dir, ofp, resname 
    
    def _get_ofp(self,
                  dkey='',
                 fancy_name=None,
                 out_dir=None,
                 resname=None,
                 ext='.tif'):
        if out_dir is None:
            out_dir=self.out_dir
        
        if resname is None:
            resname=self._get_resname(dkey, fancy_name)
            
        return os.path.join(out_dir, resname+ext)
    
    def _get_resname(self,
                     dkey='',
                     fancy_name=None,
                     ):
        if fancy_name is None:
            fancy_name=self.fancy_name
            
        return '%s_%s'%(fancy_name, dkey)
        
    def __enter__(self):
        return self
    
    def __exit__(self,  *args,**kwargs):
        pass
        """not needed by lowest object
        super().__exit__(*args, **kwargs)"""
    


        
class LogSession(Basic):
    """Session logger handling"""  
    
    def __init__(self,
 
                 wrk_dir=None,
                logfile_duplicate=False, 
                logger=None,
                logcfg_file=None,
                **kwargs):
        """
        
        Parameters
        ---------------
        logfile_duplicate : bool, default True
            Duplicate the logger into the output directory
        
        """
        if logger is None:
            
            if logcfg_file is None:
                from definitions import logcfg_file
                                
            logger = self.from_cfg_file(logcfg_file=logcfg_file, out_dir=wrk_dir)
            
            self.log_handlers(logger=logger)
 
        #=======================================================================
        # init cascase
        #=======================================================================
        super().__init__(logger=logger, wrk_dir=wrk_dir,**kwargs)
        #=======================================================================
        # duplicate logger
        #=======================================================================
        if logfile_duplicate:
            from hp.logr import get_new_file_logger
            get_new_file_logger(
                fp=os.path.join(self.out_dir, '%s_%s.log'%(
                    self.fancy_name, datetime.datetime.today().strftime('%m%d.%H.%M'))),
                logger=self.logger)
    
    def from_cfg_file(self,

            logcfg_file =None,
            out_dir=None,
            ):
        """
        creates a log file (according to the logger.conf parameters) in the passed working directory
        
        Parameters
        -------
        out_dir: str, default os.path.expanduser('~')
            location to output log files to. defaults ot 
        """
        if out_dir is None: out_dir = os.path.expanduser('~')
        if not os.path.exists(out_dir):os.makedirs(out_dir)
        #===============================================================================
        # FILE SETUP
        #===============================================================================
        if logcfg_file is None:
            #todo: check if there is a definitions file
            """create a definitions file in your project"""
            from definitions import logcfg_file #import from the definitions file
 
        assert os.path.exists(logcfg_file), 'No logger Config File found at: \n   %s'%logcfg_file
        assert logcfg_file.endswith('.conf')
        #===========================================================================
        # build logger
        #===========================================================================
        
        logger = logging.getLogger() #get the root logger
        logging.config.fileConfig(logcfg_file,
                                  defaults={'logdir':str(out_dir).replace('\\','/')},
                                  disable_existing_loggers=True,
                                  ) #load the configuration file
        'usually adds a log file to the working directory/_outs/root.log'
        logger.info('root logger initiated and configured from file: %s'%(logcfg_file))
        
        return logger
        
        
    def log_handlers(self, #convenience to readout handler info
                     logger=None):
        if logger is None:
            logger=self.logger
            
        #=======================================================================
        # #collect handler info
        #=======================================================================
        res_lib = dict()
        for handler in logger.handlers:
            
            htype = type(handler).__name__
            
            d = {'htype':htype}
            
            if 'FileHandler' in htype:
                d['filename'] = handler.baseFilename
            
            res_lib[handler.get_name()] = d
            
        #=======================================================================
        # #log
        #=======================================================================
        #get fancy string
        txt = pprint.pformat(res_lib, width=30, indent=0, compact=True, sort_dicts =False)
        
        for c in ['{', '}']: 
            txt = txt.replace(c, '') #clear some unwanted characters..
        
        logger.info('logger configured w/ %i handlers\n%s'%(len(res_lib), txt))
        
        return res_lib
    
 
    
    def __exit__(self,  *args,**kwargs):
 
        #close teh loggers
        for handler in self.logger.handlers:
            handler.close()
            
        #logging.shutdown()
            
        super().__exit__(*args, **kwargs)
        
                 
class Session(LogSession): #analysis with flexible loading of intermediate results
    """
    Basic for global methods and parameters
    
    Notes
    ------------
    typically we only instance this once
        but tests will instance multiple times
        so beware of setting containers here"""
        
    #useful for buildling a Basic object w/o the session (mostly testing)
    """TODO: integrate with init"""
    default_kwargs = dict(overwrite=True,prec=2,relative=False,write=True, 
                          proj_name=src_name, 
                          fancy_name='fancy_name', 
                          run_name='r1', 
                          obj_name='Session',
                          wrk_dir = os.path.expanduser('~'),
                          out_dir=os.path.join(os.path.expanduser('~'), 'py', 'oop', 'outs'),
                          tmp_dir=os.path.join(os.path.expanduser('~'), 'py', 'oop', 'tmp'),
                          base_dir=os.path.join(os.path.expanduser('~'), 'py', 'oop', 'outs'), 
                          ) 
    
    def __init__(self, 
                 #Session names
                 proj_name = None, fancy_name=None, run_name='r1', obj_name='Session',
                 
                 #Session directories
                 out_dir = None,wrk_dir=None,tmp_dir=None,  base_dir=None,
 
                **kwargs):
        """
        Init the session
        
        Parameters
        ------------
        see Basic
        
        wrk_dir: str, default os.path.expanduser('~')
            Base directory of the project. Used for generating default directories.            
        out_dir : str, optional
            Directory used for outputs. Defaults to a sub-directory of wrk_dir            
        tmp_dir: str, optional
            Directory for temporary outputs (i.e., cache). Defaults to a sub-directory of out_dir.
        base_dir: str, optional
            for relative=True, this is the base path
 
        """

        #=======================================================================
        # precheck
        #=======================================================================
        assert isinstance(run_name, str)
 
        #=======================================================================
        # defaults
        #=======================================================================
        if proj_name is None:            
            proj_name=src_name
            
        kwargs['proj_name']=proj_name
            
        if fancy_name is None:
            fancy_name = '%s_%s_%s'%(proj_name, run_name,  datetime.datetime.now().strftime('%m%d'))
        kwargs['fancy_name']=fancy_name
 
        #work directory
        if wrk_dir is None:
            from definitions import wrk_dir
        kwargs['wrk_dir']=wrk_dir
        
        #base directory
        if base_dir is None:
            base_dir = os.path.join(wrk_dir, 'outs', proj_name, run_name, today_str)
        if not os.path.exists(base_dir):os.makedirs(base_dir)
        kwargs['base_dir']=base_dir
            
        #output directory
        if out_dir is None:
            out_dir = base_dir
        kwargs['out_dir']=out_dir
        
        if tmp_dir is None:
            tmp_dir = os.path.join(out_dir, 'temp_%s_%s'%(
                obj_name, datetime.datetime.now().strftime('%M%S')))
            
        try:
            if os.path.exists(tmp_dir):
                delete_dir(tmp_dir)            
            os.makedirs(tmp_dir)
        except Exception as e:
            print('failed to init tmp_dir w/ \n    %s'%e)
            
        kwargs['tmp_dir']=tmp_dir
        

        
 
        #=======================================================================
        # init cascade
        #=======================================================================
        super().__init__(obj_name=obj_name,run_name=run_name,
                          **kwargs)
        
        self.logger.debug('finished Session.__init__')
        
    #===========================================================================
    # def _write_meta(self, meta_lib, **kwargs):
    #     """write a dict of dicts to a spreadsheet"""
    #     log, tmp_dir, out_dir, ofp, resname = self._func_setup('meta', subdir=False,ext='.xls',  **kwargs)
    #     
    #     #write dict of dicts to frame
    #     with pd.ExcelWriter(ofp, engine='xlsxwriter') as writer:
    #         for tabnm, d in meta_lib.items():
    #             pd.Series(d).to_frame().to_excel(writer, sheet_name=tabnm, index=True, header=True)
    #     
    #     log.info(f'wrote meta (w/ {len(meta_lib)}) to \n    {ofp}')
    #     
    #     return ofp
    #===========================================================================
    
    def _write_meta(self, meta_lib, **kwargs):
        """write a dict of dicts to a spreadsheet
        
        handles any number of nests
            not sure how flexible this is... after 2 levels I think it just dumps to a string
        
        
        
        """
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('meta', subdir=False,ext='.xls',  **kwargs)
        from hp.pd import nested_dict_to_dx
        
        #convert to simple {tabn:dataframe}\
        res_d = dict()
        for k0, d0 in meta_lib.items():
            #print(dstr(d0))
            res_d[k0] = dict_to_df(d0)
            """
            view(dict_to_df(d0))
            """
 
        
        
        #write dict of dicts to frame
        with pd.ExcelWriter(ofp, engine='xlsxwriter') as writer:
            for tabnm, dx in res_d.items():                
                dx.to_excel(writer, sheet_name=tabnm, index=True, header=True)
        
        log.info(f'wrote meta (w/ {len(meta_lib)}) to \n    {ofp}')
        
        return ofp
    
    def _write_pick(self, data, overwrite=None, **kwargs):
        """dump data to a pickle"""
        #=======================================================================
        # defaults
        #=======================================================================
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('w', subdir=False,ext='.pkl',  **kwargs)
        if overwrite is None:
            overwrite=self.overwrite
            
        #=======================================================================
        # jprecheck
        #=======================================================================
        if os.path.exists(ofp):
            assert overwrite
        
        #=======================================================================
        # write
        #=======================================================================
        with open(ofp, 'wb') as handle:
            pickle.dump(data, handle)
            
        #=======================================================================
        # wrap
        #=======================================================================
        log.info(f'wrote {type(data)} to \n    {ofp}')
        return ofp
    
    def _relpath(self, fp):
        """intelligently convert a filepath to relative"""
        if self.relative:
            assert os.path.exists(self.base_dir), self.base_dir
            assert os.path.exists(fp), fp
            #assert str(self.base_dir) in fp
            try:
                return os.path.relpath(fp, self.base_dir)
            except Exception as e:
                """only works if the root directories are the same"""
                raise IOError(f'failed to get relpath on\n    {fp}\n    {self.base_dir}\n{e}')
        else:
            return fp

def dict_to_df(d):
    """compress an arbitrarily nested dictionary to a multindex frame"""
    
    
    try:
        return pd.DataFrame.from_dict(d, orient='index').stack()
    except:
        res_d = dict()    
        for k0, v0 in d.items():
            if isinstance(v0, dict):
                res_d[k0] = dict_to_df(v0)
            else:
                res_d[k0] = pd.Series({k0:str(v0)})
            
        return pd.concat(res_d)
            
 
        
            
    
    #===========================================================================
    # df = pd.DataFrame.from_dict(d1, orient='index')
    # df.index = pd.MultiIndex.from_tuples(df.index)
    # return df
    #===========================================================================

 
    
