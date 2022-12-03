'''
Created on Mar. 26, 2020

@author: cefect

usually best to call this before any standard imports
    some modules have auto loggers to the root loger
    calling 'logging.getLogger()' after these configure will erase these
'''
import os, logging, logging.config, pprint, sys





        
def get_new_file_logger(
        logger_name='log',
        level=logging.DEBUG,
        fp=None, #file location to log to
        logger=None,
        ):
    
    #===========================================================================
    # configure the logger
    #===========================================================================
    if logger is None:
        logger = logging.getLogger(logger_name)
        
    logger.setLevel(level)
    
    #===========================================================================
    # configure the handler
    #===========================================================================
    assert fp.endswith('.log')
    
    formatter = logging.Formatter('%(levelname)s.%(name)s.%(asctime)s:  %(message)s')        
    handler = logging.FileHandler(fp, mode='w') #Create a file handler at the passed filename 
    handler.setFormatter(formatter) #attach teh formater object
    handler.setLevel(level) #set the level of the handler
    
    logger.addHandler(handler) #attach teh handler to the logger
    
    logger.info('built new file logger  here \n    %s'%(fp))
    
    return logger
    
    
def get_new_console_logger(
        logger_name='log',
        level=logging.DEBUG,
 
        logger=None,
        ):
    
    #===========================================================================
    # configure the logger
    #===========================================================================
    if logger is None:
        logger = logging.getLogger(logger_name)
        
    logger.setLevel(level)
    
    #===========================================================================
    # configure the handler
    #===========================================================================
 
    
    formatter = logging.Formatter('%(levelname)s.%(name)s:  %(message)s')        
    handler = logging.StreamHandler(
        stream=sys.stdout, #send to stdout (supports colors)
        ) #Create a file handler at the passed filename 
    handler.setFormatter(formatter) #attach teh formater object
    handler.setLevel(level) #set the level of the handler
    
    logger.addHandler(handler) #attach teh handler to the logger
    
    logger.info('built new console logger')
    
    return logger
    
    
    
    
    
    
    
    