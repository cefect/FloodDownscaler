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


import os, sys, datetime, gc, copy

from hp.dirz import delete_dir

from hp.exceptions import Error


 

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
        
        #verison info
        
        log.info('main python version: \n    %s'%sys.version)
        import numpy as np
        log.info('numpy version: %s'%np.__version__)
        import pandas as pd
        log.info('pandas version: %s'%(pd.__version__))
        
        #directory info
        log.info('os.getcwd: %s'%os.getcwd())
        
        log.info('exe: %s'%sys.executable)

        #systenm paths
        log.info('system paths')
        for k in sys.path: 
            log.info('    %s'%k)
            
            
            
    
    def __enter__(self):
        return self
    
    def __exit__(self, #destructor
             *args,**kwargs):
        
        #print('opp.__exit__ on \'%s\''%self.__class__.__name__)
        

        
        #gc.collect()
        #=======================================================================
        # #remove temporary files
        #=======================================================================
        """this fails pretty often... python doesnt seem to want to let go"""
        for fp in self.trash_fps:
            if not os.path.exists(fp): continue #ddeleted already
            try:
                if os.path.isdir(fp):
                    delete_dir(fp)
                else:
                    os.remove(fp)
                #print('    deleted %s'%fp)
            except Exception as e:
                pass
                #print('failed to delete \n    %s \n    %s'%(fp, e))
        
        #=======================================================================
        # remporary directory
        #=======================================================================
        try:
            delete_dir(self.temp_dir)
        except:
            pass
        
    
        #clear all my attriburtes
        for k in copy.copy(list(self.__dict__.keys())):
            if not k=='trash_fps':
                del self.__dict__[k]
        
        
                
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    