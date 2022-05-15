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
    
    def __exit__(self,  *args,**kwargs):
        

    
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
                 compiled_fp_d = None, #container for compiled (intermediate) results {dkey:filepath}
                 data_retrieve_hndls=None, #data retrival handles
                             #default handles for building data sets {dkey: {'compiled':callable, 'build':callable}}
                            #all callables are of the form func(**kwargs)
                            #see self._retrieve2()
                            
                wrk_dir=None, #output for working/intermediate files
                write=True, 
                exit_summary=False, #whether to write the exit summary on close

                **kwargs):
        
        """        
        data_retrieve_hndls = { **{your_handles}, **{inherited_handles}} #overwrites inherited w/ yours
        """
        assert isinstance(data_retrieve_hndls, dict), 'must past data retrival handles'
        
        
        super().__init__(**kwargs)
        
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
        log = logger.getChild('ret')
        

        start = datetime.datetime.now()
        #=======================================================================
        # 1.alredy loaded
        #=======================================================================
        """
        self.data_d.keys()
        """
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
        
        log.info('wrote %i to \n    %s'%(len(data), out_fp))
            
        
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
    
    def get_exit_summary(self,
                         ):
        

        
        #===================================================================
        # assembel summary sheets
        #===================================================================
        #merge w/ retrieve data
        for k, sub_d in self.dk_meta_d.items():
            if len(sub_d)==0:continue
            retrieve_df = pd.Series(sub_d).to_frame()
            if not k in self.smry_d:
                self.smry_d[k] = retrieve_df
            else:
                self.smry_d[k] = self.smry_d[k].reset_index().append(
                        retrieve_df.reset_index(), ignore_index=True).set_index('index')
                        
                        
        return {**{'_smry':pd.Series(self.meta_d, name='val').to_frame(),
                          '_smry.dkey':pd.DataFrame.from_dict(self.dk_meta_d).T},
                        **self.smry_d}
    
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
        if cnt>0:
            with open(os.path.join(self.out_dir, 'exit_%s.txt'%self.longname), 'a') as f:
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
            ofp = os.path.join(self.out_dir, self.longname+'__exit__%s.xls'%(
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
    
    
    
    
    
    