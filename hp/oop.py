'''
Methods for object-oriented-programming

Notes
---------------
#===============================================================================
# INHERITANCE
#===============================================================================
I've spent far too many weeks of my life strugglig with inheritance
    seems too difficult to balance simplicity, flexibility, and functionality
    
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
#from qgis.core import QgsMapLayer
from hp.dirz import delete_dir
from hp.basic import today_str
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
                 
                 #names/labels
                 proj_name      = None,  
                 run_name       = None,
                 obj_name       = None,   
                 fancy_name     = None,
                 
                 #inheritancee
                 session        = None,
                 subdir         = False,
                 
                 #controls
                 prec           = None,
                 overwrite      = None,  
                 relative       = None,  
                 write          = None,
                 
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

        prec: int, default 2
            Default float precision.
        overwrite: bool, default False
            Default behavior when attempting to overwrite a file
        relative: bool, default False
            Default behavior of filepaths (relative vs. absolute)

        session: scripts.Session, optional
            Reference to parent session
            
        subdir: bool, default False
            whether to create subidrectories (in the session defaults) using obj_name
        
        """
        
        #=======================================================================
        # personal
        #=======================================================================
 
        self.start = datetime.datetime.now()
        self.today_str = today_str
        
        #=======================================================================
        # basic attachments
        #=======================================================================
        self.session=session
        
        pars_d = dict()
        def inherit(attVal, attName, directory=False, typeCheck=None, subdir=False):
            #get from session
            if attVal is None:
                assert not session is None, 'for \'%s\' passed None but got no session'%attName
                attVal = getattr(session, attName)
                
            #make sub directories
            if subdir and directory:
                attVal = os.path.join(attVal, obj_name)
            elif subdir:
                attVal = attVal + '_' + obj_name
                
            #attach
            assert not attVal is None, attName
            setattr(self, attName, attVal)
            pars_d[attName] = attVal
            
            #handle directories
            if directory:
                if not os.path.exists(attVal):
                    os.makedirs(attVal)
            
            #check
            if not typeCheck is None:
                assert isinstance(getattr(self, attName), typeCheck), \
                    'bad type on \'%s\': %s'%(attName, type(getattr(self, attName)))
            
 
        #=======================================================================
        # basic inheritance
        #=======================================================================
        inherit(overwrite,  'overwrite', typeCheck=bool)
        inherit(prec,       'prec', typeCheck=int)
        inherit(relative,   'relative', typeCheck=bool)
        inherit(write,      'write', typeCheck=bool)
        inherit(proj_name,  'proj_name', typeCheck=str)
        inherit(run_name,   'run_name', typeCheck=str)
        inherit(fancy_name, 'fancy_name', typeCheck=str, subdir=subdir)
 
        inherit(wrk_dir,    'wrk_dir', directory=True)
        inherit(out_dir,    'out_dir', directory=True, subdir=subdir)
        inherit(tmp_dir,    'tmp_dir', directory=True, subdir=subdir)
 
        if obj_name is None:
            obj_name = self.__class__.__name__
        self.obj_name=obj_name
 
 
        #=======================================================================
        # #setup the logger
        #=======================================================================
        """TODO: add this to inheritance?"""
        if logger is None:
 
            if not session is None:
                logger=session.logger.getChild(obj_name)
            else:
                logger = logging.getLogger(obj_name)
            
        self.logger=logger
        
 

        #=======================================================================
        # wrap
        #=======================================================================
            
        #self._install_info()
        self.init_pars_d=pars_d
        self.logger.debug('finished Basic.__init__ w/\n    %s '%pars_d)
        
    
    def _func_setup(self, dkey, 
                    logger=None, out_dir=None, tmp_dir=None,ofp=None,
 
                    write=None,layname=None,ext='.tif',
                    subdir=False,
                    ):
        """common function default setup
        
        Parameters
        ----------
        subdir: bool, default False
            build the out_dir as a subdir of the default out_dir (using dkey)
            
            
        TODO
        ----------
        setup so we can inherit additional parameters from parent classes
        
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
        if write is None: write=self.write
        
        if layname is None:layname = '%s_%s'%(self.fancy_name, dkey)
         
        if ofp is None:
            if write:            
                ofp = os.path.join(out_dir, layname+ext)            
            else:
                ofp=os.path.join(tmp_dir, layname+ext)
            
        if os.path.exists(ofp):
            assert self.overwrite
            os.remove(ofp)
 
            
        return log, tmp_dir, out_dir, ofp, layname, write


            
    def _get_meta(self, keys=None,):
        if keys is None: keys = self.init_pars_d.keys()
        
        d = {k:getattr(self, k) for k in keys}
 
        
        return d
    
    def __enter__(self):
        return self
    
    def __exit__(self,  *args,**kwargs):
 
        #clear all my attriburtes
        for k in copy.copy(list(self.__dict__.keys())):
            if not k=='trash_fps':
                del self.__dict__[k]
        
                
class Session(Basic): #analysis with flexible loading of intermediate results
    """
    Project session worker for global methods and parameters
    
    Notes
    ------------
    typically we only instance this once
        but tests will instance multiple times
        so beware of setting containers here"""
        
    #useful for buildling a Basic object w/o the session (mostly testing)
    """TODO: integrate with init"""
    default_kwargs = dict(overwrite=True,prec=2,relative=False,write=True, 
                          proj_name=src_name, fancy_name='fancy_name', run_name='r1', obj_name='Session',
                          wrk_dir = os.path.expanduser('~'),
                          out_dir=os.path.join(os.path.expanduser('~'), 'py', 'oop', 'outs'),
                          tmp_dir=os.path.join(os.path.expanduser('~'), 'py', 'oop', 'tmp')
                          ) 
    
    def __init__(self, 
                 #Session names
                 proj_name = None, fancy_name=None, run_name='r1', obj_name='Session',
                 
                 #Session directories
                 out_dir = None,wrk_dir=None,tmp_dir=None,
                 
                 #Basic parameters
                 overwrite=True,
                 prec=2,
                 relative=False,
                 write=True, 
                 
                 #session algos
                 bk_lib=None,         #kwargs for builder calls {dkey:kwargs}
                 compiled_fp_d = None, #container for compiled (intermediate) results {dkey:filepath}
                 data_retrieve_hndls=None, #data retrival handles
                             #default handles for building data sets {dkey: {'compiled':callable, 'build':callable}}
                            #all callables are of the form func(**kwargs)
                            #see self._retrieve2()
                            
                #run controls
                
                exit_summary=False, #whether to write the exit summary on close
                
                #logging
                logfile_duplicate=True, 
                logger=None,
                logcfg_file=None,

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
        
        logfile_duplicate : bool, default True
            Duplicate the logger into the output directory
        
        
        """
 
        #=======================================================================
        # preset
        #=======================================================================
        if data_retrieve_hndls is None: data_retrieve_hndls=dict()
        if bk_lib is None: bk_lib=dict()
        

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
 
        if wrk_dir is None:
            wrk_dir = os.path.expanduser('~')
        kwargs['wrk_dir']=wrk_dir
            
        if out_dir is None:
            out_dir = os.path.join(wrk_dir, 'outs', proj_name, run_name, today_str)
        kwargs['out_dir']=out_dir
        
        if tmp_dir is None:
            tmp_dir = os.path.join(out_dir, 'temp_%s_%s'%(
                obj_name, datetime.datetime.now().strftime('%M%S')))
            
        if os.path.exists(tmp_dir):
            delete_dir(tmp_dir)
        os.makedirs(tmp_dir)
            
        kwargs['tmp_dir']=tmp_dir
        
        #=======================================================================
        # logger
        #=======================================================================
        if logger is None:
            from hp.logr import BuildLogr
            
            if logcfg_file is None:
                from definitions import logcfg_file

            lwrkr = BuildLogr(logcfg_file = logcfg_file, out_dir=wrk_dir)
            logger=lwrkr.logger
            
        kwargs['logger']=logger
        #=======================================================================
        # init cascade
        #=======================================================================
        super().__init__(overwrite=overwrite,prec=prec,relative=relative, write=write,
                         obj_name=obj_name,run_name=run_name,
                          **kwargs)
        
        #=======================================================================
        # duplicate logger
        #=======================================================================
        if logfile_duplicate:
            from hp.logr import get_new_file_logger
            get_new_file_logger(
                fp=os.path.join(self.out_dir, '%s_%s.log'%(
                    self.fancy_name, datetime.datetime.today().strftime('%m%d.%H.%M'))),
                logger=self.logger)
        
        #=======================================================================
        # attachments
        #=======================================================================
        self.data_d = dict() #datafiles loaded this session
    
        self.ofp_d = dict() #output filepaths generated this session
        
        if compiled_fp_d is None: compiled_fp_d=dict() #something strange here
        #=======================================================================
        # retrival handles---------
        #=======================================================================
 
        self.data_retrieve_hndls=data_retrieve_hndls
        
        #check keys
        keys = self.data_retrieve_hndls.keys()
        if len(keys)>0:
            l = set(bk_lib.keys()).difference(keys)
            assert len(l)==0, 'keymismatch on bk_lib \n    %s'%l
            
            l = set(compiled_fp_d.keys()).difference(keys)
            if not len(l)==0:
                raise KeyError('keymismatch on compiled_fp_d \n    %s'%l)
            
        #attach    
        self.bk_lib=bk_lib
        self.compiled_fp_d = compiled_fp_d
        self.write=write
        self.exit_summary=exit_summary
        
        #start meta
        self.dk_meta_d = {k:dict() for k in keys}
        self.meta_d=dict()
        self.smry_d=dict()
        
    def retrieve(self,  
                 dkey,
                 *args,
                 logger=None,
                 **kwargs
                 ):
        """flexible 3 source data retrival"""
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log = logger.getChild('ret')

        start = datetime.datetime.now()
        #=======================================================================
        # 1.alredy loaded
        #=======================================================================
 
        if dkey in self.data_d:
            log.info('loading \'%s\' from data_d'%dkey)
            try:
                return copy.deepcopy(self.data_d[dkey])
            except Exception as e:
                log.warning('failed to get a copy of \"%s\'... passing raw w/ \n    %s'%(dkey, e))
                return self.data_d[dkey]
        
        #=======================================================================
        # retrieve handles
        #=======================================================================
        log.debug('loading %s'%dkey)
                
        assert dkey in self.data_retrieve_hndls, dkey
        
        hndl_d = self.data_retrieve_hndls[dkey]
        meta_d = dict()
        #=======================================================================
        # 2.compiled provided
        #=======================================================================
 
        if dkey in self.compiled_fp_d and 'compiled' in hndl_d:
            log.info('building \'%s\' from compiled'%(dkey))
            data = hndl_d['compiled'](fp=self.compiled_fp_d[dkey], dkey=dkey)
            method='loaded pre-compiled from %s'%self.compiled_fp_d[dkey]
        #=======================================================================
        # 3.build from scratch
        #=======================================================================
        else:
            assert 'build' in hndl_d, 'no build handles for %s'%dkey
            log.info('building \'%s\' from %s'%(dkey, hndl_d['build']))
            
            #retrieve builder kwargs
            if dkey in self.bk_lib:
                bkwargs=self.bk_lib[dkey].copy()
                bkwargs.update(kwargs) #function kwargs take precident
                kwargs = bkwargs
                """
                clearer to the user
                also gives us more control for calls within calls
                """

            data = hndl_d['build'](*args, dkey=dkey,logger=logger, **kwargs)
            
            method='built w/ %s'%kwargs
            
        #=======================================================================
        # store
        #=======================================================================
        assert data is not None, '\'%s\' got None'%dkey
        
        """
        if isinstance(data, QgsMapLayer):
            self.mstore.addMapLayer(data)
            
            meta_d.update({'layname':data.name(), 'source':data.source()})
            
        else:
            assert hasattr(data, '__len__'), '\'%s\' failed to retrieve some data'%dkey"""
            
        self.data_d[dkey] = data
        
        #=======================================================================
        # meta
        #=======================================================================
        tdelta = round((datetime.datetime.now() - start).total_seconds(), 1)
        meta_d.update({
            'tdelta (secs)':tdelta, 'dtype':type(data), 'method':method})
            
        self.dk_meta_d[dkey].update(meta_d)
        #=======================================================================
        # wrap
        #=======================================================================
        log.info('finished on \'%s\' w/   dtype=%s'%(dkey,  type(data)))
        
        return data
    
    def load_pick(self,
                  fp=None, 
                  dkey=None,
                  ):
        
        assert os.path.exists(fp), 'bad fp for \'%s\' \n    %s'%(dkey, fp)
        
        with open(fp, 'rb') as f:
            data = pickle.load(f)
            
        return data
    
    def write_pick(self, 
                   data, 
                   out_fp,
                   overwrite=None,
                   protocol = 3, # added in Python 3.0. It has explicit support for bytes
                   logger=None):
        
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log=logger.getChild('write_pick')
        if overwrite is None: overwrite=self.overwrite
        
        #=======================================================================
        # checks
        #=======================================================================
        
        if os.path.exists(out_fp):
            assert overwrite, out_fp
            
        assert out_fp.endswith('.pickle')
            
        log.debug('writing to %s'%out_fp)
        
        with open(out_fp,  'wb') as f:
            pickle.dump(data, f, protocol)
        
        log.info('wrote %i to \n    %s'%(len(data), out_fp))
        
        return out_fp
    
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
        
    def _get_meta(self, #get a dictoinary of metadat for this model
                 ):
        
        d = super()._get_meta()
        
        if len(self.data_d)>0:
            d['data_d.keys()'] = list(self.data_d.keys())
            
        if len(self.ofp_d)>0:
            d['ofp_d.keys()'] = list(self.ofp_d.keys())
            
        if len(self.compiled_fp_d)>0:
            d['compiled_fp_d.keys()'] = list(self.compiled_fp_d.keys())
            
        if len(self.bk_lib)>0:
            d['bk_lib'] = copy.deepcopy(self.bk_lib)
            
        return d
    
    def get_exit_summary(self,
                         ):
        
        #===================================================================
        # assembel summary sheets
        #===================================================================
        #merge w/ retrieve data
        for k, sub_d in self.dk_meta_d.items():
            if len(sub_d)==0:continue
            
            #get the meta for this key
            if isinstance(sub_d, dict):
                retrieve_df = pd.Series(sub_d).to_frame()
            elif isinstance(sub_d, pd.DataFrame):
                retrieve_df=sub_d
            else:
                raise TypeError(type(sub_d))
            
            #update the container
            if not k in self.smry_d:
                self.smry_d[k] = retrieve_df
            else:
                self.smry_d[k] = self.smry_d[k].reset_index().append(
                        retrieve_df.reset_index(), ignore_index=True).set_index('index')
                        
        return {**{'_smry':pd.Series(self.meta_d, name='val').to_frame(),
                          '_smry.dkey':pd.DataFrame.from_dict(self.dk_meta_d).T},
                        **self.smry_d}
        
    def _clear_all(self): #clear all the loaded data
        self.data_d = dict()
        self.mstore.removeAllMapLayers()
        self.compiled_fp_d.update(self.ofp_d) #copy everything over to compile
        gc.collect()
        
    def _log_datafiles(self, 
                       log=None,
                       d = None,
                       ):
        if log is None: log=self.logger
        
        if d is None:
        
            #print each datafile
            d = copy.copy(self.compiled_fp_d) #start with the passed
            d.update(self.ofp_d) #add the loaded

        s0=''
        for k,v in d.items():
            s0 = s0+'\n    \'%s\':r\'%s\','%(k,  v)
                
        log.info(s0)
    
    def __exit__(self, #destructor
                 *args, **kwargs):
        
        print('oop.Session.__exit__ (%s)'%self.__class__.__name__)
        
        #=======================================================================
        # log major containers
        #=======================================================================
        cnt=0
        msg=''
        if len(self.data_d)>0:
            msg+='\n    data_d.keys(): \n        %s'%(list(self.data_d.keys()))
            self.data_d = dict() #not necessiary any more
        
        if len(self.ofp_d)>0:
            msg+='\n    ofp_d (%i):'%len(self.ofp_d)
            for k,v in self.ofp_d.items():
                cnt+=1
                msg+='\n        \'%s\':r\'%s\','%(k,v)
            msg+='\n\n'
            self.ofp_d = dict()
              
        try:
            log = self.logger.getChild('e')
            log.info(msg)
        except:
            print(msg)
            
        #=======================================================================
        # write it
        #=======================================================================
        if cnt>0 and self.exit_summary:
            assert os.path.exists(self.tmp_dir)
            with open(os.path.join(self.tmp_dir, 'exit_%s.txt'%self.fancy_name), 'a') as f:
                f.write(datetime.datetime.now().strftime('%H%M%S'))
                f.write(msg)
        #=======================================================================
        # extneded exit summary
        #=======================================================================
        """copied over from RICorDE... need to test"""
        if self.exit_summary: 
            
            tdelta = datetime.datetime.now() - self.start
            runtime = tdelta.total_seconds()/60.0
            
            self.meta_d = {**{'now':datetime.datetime.now(), 'runtime (mins)':runtime}, **self.meta_d}
            
            smry_d = self.get_exit_summary()
            
            #=======================================================================
            # write the summary xlsx
            #=======================================================================
    
            #get the filepath
            ofp = os.path.join(self.out_dir, self.fancy_name+'__exit__%s.xls'%(
                datetime.datetime.now().strftime('%H%M%S')))
            if os.path.exists(ofp):
                assert self.overwrite
                os.remove(ofp)
        
            #write
            try:
                with pd.ExcelWriter(ofp) as writer:
                    for tabnm, df in smry_d.items():
                        df.to_excel(writer, sheet_name=tabnm, index=True, header=True)
                        
                print('wrote %i summary sheets to \n    %s'%(len(smry_d), ofp))
                    
            except Exception as e:
                print('failed to write summaries w/ \n    %s'%e)
        
            #=======================================================================
            # wrap
            #=======================================================================
            self.logger.info('finished in %.2f mins'%(runtime))
        
        super().__exit__(*args, **kwargs)
    
