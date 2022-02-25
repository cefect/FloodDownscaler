'''
Created on Mar 10, 2019

@author: cef

object oriented programming

#===============================================================================
# INHERITANCE
#===============================================================================
I've spent far too many weeks of my life strugglig with inheritance
    seems to difficult to balance simplicity, flexibility, and functionality
    
2021-10-31: settled on top down control
    force the caller object to first extract any attributes they want to pass down
    then run these through the childs init
    i.e., the child is agnostic to the source of the attribute
    this keeps the majority of scripts simple
        scripts that want to get complicated with inheritance can do this at the caller level
    
    using the 'inher_d'  to store 'object adding the requirement':[attn] the object to use if it wants to spawn children
        see get_inher_atts
'''


import os, sys, datetime, gc, copy, pickle, pprint

from hp.dirz import delete_dir

from hp.exceptions import Error

import numpy as np
import pandas as pd
 
 

#===============================================================================
# functions------------------------------------------------------------------- 
#===============================================================================

class Basic(object): #simple base class
    def __init__(self, 

                 
                 #directories
                 out_dir        = None,
                 temp_dir       = None,
                 work_dir       = r'C:\LS\10_OUT\coms',
                 
                 #names/labels
                 name           = None, #task or function-based name ('e.g., Clean). nice to capitalize
                 tag            = None, #session or run name (e.g., 0402) 
                 longname       = None,
                 
                 #inheritancee
                 session        = None,
                 
                 #controls
                 prec           = 2,
                 overwrite      = False, #file overwriting control
                 
                 logger         = None,
                 
                 ):
        
        
        #=======================================================================
        # personal
        #=======================================================================
        self.trash_fps = list() #container for files to delete on exit
        self.start = datetime.datetime.now()
        self.today_str = datetime.datetime.today().strftime('%Y%m%d')
        
        #=======================================================================
        # basic attachments
        #=======================================================================
        self.session=session
        self.work_dir=work_dir
 
        self.overwrite=overwrite
        self.prec=prec
        
        #=======================================================================
        # complex attachments
        #=======================================================================
        if name is None:
            name = self.__class__.__name__
        self.name=name
        
 
 
 
 
        # run tag
        if tag is None:
            tag = 't'+datetime.datetime.today().strftime('%H%M')
            
        self.tag=tag
 
        # labels
        if longname is None:
            longname = '%s_%s_%s'%(self.name, self.tag,  datetime.datetime.now().strftime('%m%d'))
                
        self.longname = longname
 

        #=======================================================================
        # output directory
        #=======================================================================
        if out_dir is None:
            if not tag == '':
                out_dir = os.path.join(work_dir, 'outs', name, tag, self.today_str)
            else:
                out_dir = os.path.join(work_dir, 'outs', name, self.today_str)
                

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
                
        self.out_dir=out_dir 
        
        #=======================================================================
        # #temporary directory
        #=======================================================================
        """not removing this automatically"""
        if temp_dir is None:
 
            temp_dir = os.path.join(self.out_dir, 'temp_%s_%s'%(
                self.__class__.__name__, datetime.datetime.now().strftime('%M%S')))
            
            if os.path.exists(temp_dir):
                delete_dir(temp_dir)
 
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
            
        self.temp_dir = temp_dir
        
        #=======================================================================
        # #setup the logger
        #=======================================================================
        if logger is None:
 
            if not session is None:
                logger=session.logger.getChild(name)
            else:
                
                os.chdir(work_dir) #set this to the working directory
                print('working directory set to \"%s\''%os.getcwd())
            
                from hp.logr import BuildLogr
                lwrkr = BuildLogr()
                logger=lwrkr.logger
                lwrkr.duplicate(self.out_dir, 
                            basenm='%s_%s'%(tag, datetime.datetime.today().strftime('%m%d.%H.%M')))

            
        self.logger=logger

        #=======================================================================
        # wrap
        #=======================================================================
 
            
        #self._install_info()
        
        self.logger.debug('finished Basic.__init__ ')
        

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
            
    def _get_meta(self):
        attns = ['tag', 'name', 'longname', 'start', 'today_str', 'prec', 'work_dir', 'out_dir']
        
        d = {k:getattr(self, k) for k in attns}
        d = {**d, **{'python':sys.version, 'numpy':np.__version__, 'pandas':pd.__version__}}
        
        return d
            
            
            
    
    def __enter__(self):
        return self
    
    def __exit__(self, #destructor
             *args,**kwargs):
        

    
        #clear all my attriburtes
        for k in copy.copy(list(self.__dict__.keys())):
            if not k=='trash_fps':
                del self.__dict__[k]
        
        
                
class Session(Basic): #analysis with flexible loading of intermediate results
    """typically we only instance this once
        but tests will instance multiple times
        so beware of setting containers here"""

    
    
    
    def __init__(self, 
                 bk_lib=dict(),         #kwargs for builder calls {dkey:kwargs}
                 compiled_fp_d = dict(), #container for compiled (intermediate) results {dkey:filepath}
                 data_retrieve_hndls=None, #data retrival handles
                             #default handles for building data sets {dkey: {'compiled':callable, 'build':callable}}
                            #all callables are of the form func(**kwargs)
                            #see self._retrieve2()
                            
                wrk_dir=None, #output for working/intermediate files
                write=True,

                **kwargs):
        
        assert isinstance(data_retrieve_hndls, dict), 'must past data retrival handles'
        
        
        super().__init__(**kwargs)
        
        self.data_d = dict() #datafiles loaded this session
    
        self.ofp_d = dict() #output filepaths generated this session
        
        
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
            assert len(l)==0, 'keymismatch on compiled_fp_d \n    %s'%l
            
            
        #attach    
        self.bk_lib=bk_lib
        self.compiled_fp_d = compiled_fp_d
        self.write=write
        
        
        #start meta
        self.dk_meta_d = {k:dict() for k in keys}
 
            
        
        #=======================================================================
        # defaults
        #=======================================================================
        if wrk_dir is None:
            wrk_dir = os.path.join(self.out_dir, 'working')
        
        if not os.path.exists(wrk_dir):
            os.makedirs(wrk_dir)
            
        self.wrk_dir = wrk_dir
        
        
    def retrieve(self, #flexible 3 source data retrival
                 dkey,
                 *args,
                 logger=None,
                 **kwargs
                 ):
        
        if logger is None: logger=self.logger
        log = logger.getChild('retrieve')
        

        start = datetime.datetime.now()
        #=======================================================================
        # 1.alredy loaded
        #=======================================================================
        """
        self.data_d.keys()
        """
        if dkey in self.data_d:
            try:
                return copy.deepcopy(self.data_d[dkey])
            except Exception as e:
                log.warning('failed to get a copy of \"%s\' w/ \n    %s'%(dkey, e))
                return self.data_d[dkey]
            
        
        #=======================================================================
        # retrieve handles
        #=======================================================================
        log.info('loading %s'%dkey)
                
        assert dkey in self.data_retrieve_hndls, dkey
        
        hndl_d = self.data_retrieve_hndls[dkey]
        
        #=======================================================================
        # 2.compiled provided
        #=======================================================================
 
        if dkey in self.compiled_fp_d and 'compiled' in hndl_d:
            data = hndl_d['compiled'](fp=self.compiled_fp_d[dkey], dkey=dkey)
            method='loaded pre-compiled from %s'%self.compiled_fp_d[dkey]
        #=======================================================================
        # 3.build from scratch
        #=======================================================================
        else:
            assert 'build' in hndl_d, 'no build handles for %s'%dkey
            
            #retrieve builder kwargs
            if dkey in self.bk_lib:
                bkwargs=self.bk_lib[dkey].copy()
                bkwargs.update(kwargs) #function kwargs take precident
                kwargs = bkwargs
                """
                clearer to the user
                also gives us more control for calls within calls
                """

            data = hndl_d['build'](*args, dkey=dkey, **kwargs)
            
            method='built w/ %s'%kwargs
            
        #=======================================================================
        # store
        #=======================================================================
        assert data is not None, '\'%s\' got None'%dkey
        assert hasattr(data, '__len__'), '\'%s\' failed to retrieve some data'%dkey
        self.data_d[dkey] = data
        
        tdelta = round((datetime.datetime.now() - start).total_seconds(), 1)
            
        self.dk_meta_d[dkey].update({
            'tdelta (secs)':tdelta, 'dtype':type(data), 'len':len(data), 'method':method})
        #=======================================================================
        # wrap
        #=======================================================================
        log.info('finished on \'%s\' w/ len=%i dtype=%s'%(dkey, len(data), type(data)))
        
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
        
        log.info('wrote %i to %s'%(len(data), out_fp))
            
        
        return out_fp
        
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
    
    def __exit__(self, #destructor
                 *args, **kwargs):
        
        print('oop.Session.__exit__ (%s)'%self.__class__.__name__)
        
        #=======================================================================
        # log major containers
        #=======================================================================
        if len(self.data_d)>0:
            print('    data_d.keys(): %s'%(list(self.data_d.keys())))
            self.data_d = dict() #not necessiary any more
        
        if len(self.ofp_d)>0:
            print('    ofp_d (%i):'%len(self.ofp_d))
            for k,v in self.ofp_d.items():
                print('        \'%s\':r\'%s\','%(k,v))
            print('\n')
            self.ofp_d = dict()
              
              
        
        
        super().__exit__(*args, **kwargs)
    
    
    
    
    
    