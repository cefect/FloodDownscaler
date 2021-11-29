'''
Created on Mar. 26, 2020

@author: cefect

usually best to call this before any standard imports
    some modules have auto loggers to the root loger
    calling 'logging.getLogger()' after these configure will erase these
'''
import os, logging, logging.config




class BuildLogr(object): #simple class to build a logger
    
    def __init__(self,
            work_dir, 
            logcfg_file =r'C:\LS\03_TOOLS\coms\logger.conf',
            ):
        """
        creates a log file (according to the logger.conf parameters) in the passed working directory
        """
        #===============================================================================
        # FILE SETUP
        #===============================================================================
        assert os.path.exists(work_dir), work_dir
        os.chdir(work_dir) #set this to the working directory
        print('working directory set to \"%s\''%os.getcwd())
        
        assert os.path.exists(logcfg_file), 'No logger Config File found at: \n   %s'%logcfg_file
    
        #===========================================================================
        # build logger
        #===========================================================================
        
        logger = logging.getLogger() #get the root logger
        
 
        
        logging.config.fileConfig(logcfg_file) #load the configuration file
        'usually adds a log file to the working directory/_outs/root.log'
        logger.info('root logger initiated and configured from file: %s'%(logcfg_file))
        
        
        
        
        self.logger = logger
        
        
        
    def duplicate(self, #duplicate the root logger to a diretory
                  out_dir, #directory to place the new logger
                  basenm = 'duplicate', #basename for the new logger file
                  level = logging.DEBUG,
                  ):
        
        #===============================================================================
        # # Load duplicate log file
        #===============================================================================
        assert os.path.exists(out_dir)
        logger_file_path = os.path.join(out_dir, '%s.log'%basenm)
        
        #build the handler
        formatter = logging.Formatter('%(asctime)s.%(levelname)s.%(name)s:  %(message)s')        
        handler = logging.FileHandler(logger_file_path, mode='w') #Create a file handler at the passed filename 
        handler.setFormatter(formatter) #attach teh formater object
        handler.setLevel(level) #set the level of the handler
        
        self.logger.addHandler(handler) #attach teh handler to the logger
        
        self.logger.info('duplicate logger \'level = %i\' built: \n    %s'%(
            level, logger_file_path))