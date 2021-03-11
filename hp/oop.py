'''
Created on Mar 10, 2019

@author: cef

object oriented programming

'''


import os, copy, sys, time, re, logging



from hp.exceptions import Error


mod_logger = logging.getLogger(__name__)
mod_logger.debug('initialized')

#===============================================================================
# functions------------------------------------------------------------------- 
#===============================================================================

class Basic(object): #simple base class
    
    def __init__(self, 
                 logger         = None,
                 out_dir        = None,
                 work_dir       = r'C:\LS\03_TOOLS\misc',
                 mod_name       = 'Simp',
                 tag            = '',
                 prec           = 2,
                 overwrite      = False, #file overwriting control
                 ):
        
        
        
        
        """
        logger.info('test')
        """
        self.work_dir = work_dir
        self.mod_name = mod_name
        self.tag = tag
        self.prec=prec
        self.overwrite=overwrite
            
        #=======================================================================
        # output directory
        #=======================================================================
        if out_dir is None:
            out_dir = os.path.join(work_dir, 'outs')
            
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
            
        self.out_dir = out_dir
        
        #=======================================================================
        # #setup the logger
        #=======================================================================
        if logger is None:
            from hp.logr import BuildLogr
            lwrkr = BuildLogr(work_dir)
            logger=lwrkr.logger

            
        self.logger=logger
            
        
        
        self.logger.debug('finished Basic.__init__')
        
    def get_install_info(self,
                         log = None): #print version info
        if log is None: log = self.logger
        
        #verison info
        
        self.logger.info('main python version: \n    %s'%sys.version)
        import numpy as np
        self.logger.info('numpy version: %s'%np.__version__)
        import pandas as pd
        self.logger.info('pandas version: %s'%(pd.__version__))
        
        #directory info
        self.logger.info('os.getcwd: %s'%os.getcwd())
        
        log.info('exe: %s'%sys.executable)

        #systenm paths
        log.info('system paths')
        for k in sys.path: 
            log.info('    %s'%k)
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    