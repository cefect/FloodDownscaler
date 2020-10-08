'''
Created on Mar 10, 2019

@author: cef

object oriented programming


py2.7
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
                 logger         = mod_logger,
                 out_dir        = None,
                 work_dir       = r'C:\LS\02_WORK\02_Mscripts\CoC_backcast\03_SOFT\03_py',
                 mod_name       = 'Simp',
                 tag            = 'BasicTag',
                 prec           = 2,
                 ):
        
        
        self.logger=logger
        self.work_dir = work_dir
        self.mod_name = mod_name
        self.tag = tag
        self.prec=prec
            
        self.out_dir = out_dir
        
        
        self.logger.debug('finished Basic.__init__')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    