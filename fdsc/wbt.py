'''
Created on Dec. 5, 2022

@author: cefect
'''

import os, logging, sys
from whitebox_tools import WhiteboxTools
from definitions import wbt_dir
from hp.oop import Basic

 

class WBT_worker(Basic, WhiteboxTools):
    """init the WhiteboxTools to this project"""
    def __init__(self, 
                 build_dir=None,
                 logger=None,
                 compress_rasters=False,
                 **kwargs):
        
        WhiteboxTools.__init__(self)
        
        #setup logger
        """requires  a logger for the callback method"""
        if logger is None:
            #basic standalone setup
            logging.basicConfig(force=True, #overwrite root handlers
                stream=sys.stdout, #send to stdout (supports colors)
                level=logging.INFO, #lowest level to display
                )
            logger = logging.getLogger()
 
        
        
        Basic.__init__(self, logger=logger, **kwargs)
        
        #=======================================================================
        # customizet wbt
        #=======================================================================
        #set the whitebox dir
        if build_dir is None:
            build_dir = wbt_dir
            
        assert os.path.exists(build_dir)
        self.set_whitebox_dir(build_dir)
        #print(f'set_whitebox_dir({build_dir})')
        
        #callback default
        self.set_default_callback(self.__callback__)
        
        #verbosity
        if __debug__:
            assert self.set_verbose_mode(True)==0
        else:
            self.set_verbose_mode(False)
            
        assert self.set_compress_rasters(compress_rasters)==0
            

        
        #=======================================================================
        # wrap
        #=======================================================================
        self.logger.debug('setup WhiteBoxTools w/\n' +\
                 
                 f'    set_whitebox_dir({build_dir})\n'+\
                 f'    set_verbose_mode({__debug__})\n'+\
                 f'    set_compress_rasters({compress_rasters})\n'+\
                 f'    set_default_callback(self.__callback__)\n'
                 #"Version information: {}".format(self.version())                
                 )
                 
        
    def __callback__(self, value):
        """default callback methjod"""
        if not "%" in value:
            self.logger.debug(value)
        
            
        
        
        
        