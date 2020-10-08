'''
Created on Oct. 8, 2020

@author: cefect
'''



#===============================================================================
# # standard imports -----------------------------------------------------------
#===============================================================================
import time, sys, os, logging, copy, shutil, re, inspect, weakref

import numpy as np
import pandas as pd

#===============================================================================
# import QGIS librarires
#===============================================================================
from qgis.core import *
from PyQt5.QtCore import QVariant, QMetaType 
import processing  


from qgis.analysis import QgsNativeAlgorithms

"""throws depceciationWarning"""
import processing  

#===============================================================================
# custom imports
#===============================================================================

from hp.exceptions import Error

from hp.oop import Basic


#===============================================================================
# logging
#===============================================================================
mod_logger = logging.getLogger(__name__)


#===============================================================================
# funcs
#===============================================================================

class Qproj(Basic):
    """
    common methods for Qgis projects
    """
    
    def __init__(self, 
                 crs_id = 'EPSG:4326',
                 
                 **kwargs):

        
        mod_logger.debug('Qproj super')
        
        super().__init__(**kwargs) #initilzie teh baseclass
        
        
        #=======================================================================
        # setup qgis
        #=======================================================================
        self.qap = self.init_qgis()
        self.qproj = QgsProject.instance()

        self.init_algos()
        
        self.set_vdrivers()
        
        self.mstore = QgsMapLayerStore() #build a new map store
        

        self.crs = QgsCoordinateReferenceSystem(crs_id)
        
        
        
        if not self.proj_checks():
            raise Error('failed checks')
        

        
        self.logger.info('Qproj __INIT__ finished w/ crs \'%s\''%self.crs.authid())
        
        
        return
    
    
    def init_qgis(self,
                  gui = False): #hp function to instantiate the program'
        """
        WARNING: need to hold this app somewhere. call in the module you're working in (scripts)
        
        """
        log = self.logger.getChild('init_qgis')
        try:
            
            QgsApplication.setPrefixPath(r'C:/OSGeo4W64/apps/qgis', True)
            
            app = QgsApplication([], gui)
            #   Update prefix path
            #app.setPrefixPath(r"C:\OSGeo4W64\apps\qgis", True)
            app.initQgis()
            #logging.debug(QgsApplication.showSettings())
            log.info(u' QgsApplication.initQgis. version: %s, release: %s'%(
                    Qgis.QGIS_VERSION.encode('utf-8'), Qgis.QGIS_RELEASE_NAME.encode('utf-8')))
            
            return app
        except:
            log.warning('QGIS failed to initiate')
            raise SystemError
        
        
    def init_algos(self): #initiilize processing and add providers
    
        from processing.core.Processing import Processing
    
        Processing.initialize()
    
        QgsApplication.processingRegistry().addProvider(QgsNativeAlgorithms())
        
        self.logger.info('processing initilzied')
    
        
        return True
    
    
    def set_vdrivers(self):
        
        #build vector drivers list by extension
        """couldnt find a good built-in to link extensions with drivers"""
        vlay_drivers = {'SpatiaLite':'sqlite', 'OGR':'shp'}
        
        
        #vlay_drivers = {'sqlite':'SpatiaLite', 'shp':'OGR','csv':'delimitedtext'}
        
        for ext in QgsVectorFileWriter.supportedFormatExtensions():
            dname = QgsVectorFileWriter.driverForExtension(ext)
            
            if not dname in vlay_drivers.keys():
            
                vlay_drivers[dname] = ext
            
        #add in missing/duplicated
        for vdriver in QgsVectorFileWriter.ogrDriverList():
            if not vdriver.driverName in vlay_drivers.keys():
                vlay_drivers[vdriver.driverName] ='?'
                
        self.vlay_drivers = vlay_drivers
        
        self.logger.debug('built driver:extensions dict: \n    %s'%vlay_drivers)
        
        return
    
    
    def proj_checks(self): #placeholder
        #=======================================================================
        # log = self.logger.getChild('proj_checks')
        # 
        # if not self.driverName in self.vlay_drivers:
        #     raise Error('unrecognized driver name')
        # 
        # if not self.out_dName in self.vlay_drivers:
        #     raise Error('unrecognized driver name')
        # 
        # log.info('project passed all checks')
        #=======================================================================
        
        return True
    
    
    
if __name__ == '__main__':

    Qproj()
    

    
    print('finished')
    
    
    
    
    
    
    
    