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
    
    crs_id = 'EPSG:4326'
    
    driverName = 'SpatiaLite' #default data creation driver type
    out_dName = driverName #default output driver/file type
    
    def __init__(self, 
                 feedback=None, 
                 crs = None,
                 
                 **kwargs):

        
        mod_logger.debug('Qproj super')
        
        super().__init__(**kwargs) #initilzie teh baseclass
        
        
        #=======================================================================
        # setup qgis
        #=======================================================================
        if feedback is None:
            feedback = MyFeedBackQ()
        self.feedback = feedback
            
        self.qap = self.init_qgis()
        self.qproj = QgsProject.instance()

        self.init_algos()
        
        self.set_vdrivers()
        
        self.mstore = QgsMapLayerStore() #build a new map store
        

        if crs is None: 
            crs = QgsCoordinateReferenceSystem(self.crs_id)
            
        self.crs = crs
        
        
        
        if not self.proj_checks():
            raise Error('failed checks')
        

        
        self.logger.info('Qproj __INIT__ finished w/ crs \'%s\''%self.crs.authid())
        
        
        return
    
    
    def init_qgis(self, #instantiate qgis
                  gui = False): 
        """
        WARNING: need to hold this app somewhere. call in the module you're working in (scripts)
        
        """
        log = self.logger.getChild('init_qgis')
        
        try:
            
            QgsApplication.setPrefixPath(r'C:/OSGeo4W64/apps/qgis-ltr', True)
            
            app = QgsApplication([], gui)
            #   Update prefix path
            #app.setPrefixPath(r"C:\OSGeo4W64\apps\qgis", True)
            app.initQgis()
            #logging.debug(QgsApplication.showSettings())
            """ was throwing unicode error"""
            log.info(u' QgsApplication.initQgis. version: %s, release: %s'%(
                Qgis.QGIS_VERSION.encode('utf-8'), Qgis.QGIS_RELEASE_NAME.encode('utf-8')))
            return app
        
        except:
            raise Error('QGIS failed to initiate')
        
    def init_algos(self): #initiilize processing and add providers
        """
        crashing without raising an Exception
        """
    
    
        log = self.logger.getChild('init_algos')
        
        if not isinstance(self.qap, QgsApplication):
            raise Error('qgis has not been properly initlized yet')
        
        from processing.core.Processing import Processing
    
        Processing.initialize() #crashing without raising an Exception
    
        QgsApplication.processingRegistry().addProvider(QgsNativeAlgorithms())
        
        assert not self.feedback is None, 'instance needs a feedback method for algos to work'
        
        log.info('processing initilzied w/ feedback: \'%s\''%(type(self.feedback).__name__))
        

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
        
    def set_crs(self, #load, build, and set the project crs
                authid =  None):
        
        #=======================================================================
        # setup and defaults
        #=======================================================================
        log = self.logger.getChild('set_crs')
        
        if authid is None: 
            authid = self.crs_id
        
        if not isinstance(authid, int):
            raise IOError('expected integer for crs')
        
        #=======================================================================
        # build it
        #=======================================================================
        self.crs = QgsCoordinateReferenceSystem(authid)
        
        if not self.crs.isValid():
            raise IOError('CRS built from %i is invalid'%authid)
        
        #=======================================================================
        # attach to project
        #=======================================================================
        self.qproj.setCrs(self.crs)
        
        if not self.qproj.crs().description() == self.crs.description():
            raise Error('qproj crs does not match sessions')
        
        log.info('Session crs set to EPSG: %i, \'%s\''%(authid, self.crs.description()))
           
    def proj_checks(self):
        log = self.logger.getChild('proj_checks')
        
        if not self.driverName in self.vlay_drivers:
            raise Error('unrecognized driver name')
        
        if not self.out_dName in self.vlay_drivers:
            raise Error('unrecognized driver name')
        
        

        
        assert not self.feedback is None
        

        
        log.info('project passed all checks')
        
        return True
    
    def vlay_load(self,
                  file_path,

                  providerLib='ogr',


                  logger = None,
                  ):
        
        #=======================================================================
        # setup
        #=======================================================================
        if logger is None: logger=self.logger
        log = logger.getChild('vlay_load')
        
        
        #file
        assert os.path.exists(file_path)
        fname, ext = os.path.splitext(os.path.split(file_path)[1])
        
        #=======================================================================
        # defaults
        #=======================================================================

        
        #===========================================================================
        # load the layer--------------
        #===========================================================================


        vlay_raw = QgsVectorLayer(file_path,fname,providerLib)


        self.mstore.addMapLayer(vlay_raw)
            
            
        #===========================================================================
        # checks
        #===========================================================================
        if not isinstance(vlay_raw, QgsVectorLayer): 
            raise IOError
        
        #check if this is valid
        if not vlay_raw.isValid():
            raise Error('loaded vlay \'%s\' is not valid. \n \n did you initilize?'%vlay_raw.name())
        
        #check if it has geometry
        if vlay_raw.wkbType() == 100:
            raise Error('loaded vlay has NoGeometry')
        
        #check coordinate system
        if not vlay_raw.crs().isValid():
            raise Error('bad crs')
        
        if vlay_raw.crs().authid() == '':
            log.warning('bad crs')
        
        if not vlay_raw.crs() == self.crs:
            log.warning('crs: \'%s\' doesnt match project: %s'%(
                vlay_raw.crs().authid(), self.crs.authid()))
            
            
        #=======================================================================
        # wrap
        #=======================================================================
        self.createspatialindex(vlay_raw, logger=log)
        vlay = vlay_raw
        #add to project
        'needed for some algorhithims. moved to algos'
        #vlay = self.qproj.addMapLayer(vlay, False)
        

        dp = vlay.dataProvider()
                
        log.info('loaded \'%s\' as \'%s\' \'%s\'  with %i feats crs: \'%s\' from file: \n     %s'
                    %(vlay.name(), dp.storageType(), 
                      QgsWkbTypes().displayString(vlay.wkbType()), 
                      dp.featureCount(), 
                      vlay.crs().authid(),
                      file_path))
        
        return vlay
    
    
    
    def createspatialindex(self,
                     in_vlay,
                     logger = None,
                     ):

        #=======================================================================
        # presets
        #=======================================================================
        algo_nm = 'qgis:createspatialindex'
        if logger is None: logger=self.logger
        #log = logger.getChild('createspatialindex')


 
        #=======================================================================
        # assemble pars
        #=======================================================================
        #assemble pars
        ins_d = { 'INPUT' : in_vlay }
        
        #log.debug('executing \'%s\' with ins_d: \n    %s'%(algo_nm, ins_d))
        
        res_d = processing.run(algo_nm, ins_d, feedback=self.feedback)
        
        return
        
        
class MyFeedBackQ(QgsProcessingFeedback):
    """
    wrapper for easier reporting and extended progress
    
    Dialogs:
        built by QprojPlug.qproj_setup()
    
    Qworkers:
        built by Qcoms.__init__()
    
    """
    
    def __init__(self,
                 logger=mod_logger):
        
        self.logger=logger.getChild('FeedBack')
        
        super().__init__()

    def setProgressText(self, text):
        self.logger.debug(text)

    def pushInfo(self, info):
        self.logger.info(info)

    def pushCommandInfo(self, info):
        self.logger.info(info)

    def pushDebugInfo(self, info):
        self.logger.info(info)

    def pushConsoleInfo(self, info):
        self.logger.info(info)

    def reportError(self, error, fatalError=False):
        self.logger.error(error)
        
    
    def upd_prog(self, #advanced progress handling
             prog_raw, #pass None to reset
             method='raw', #whether to append value to the progress
             ): 
            
        #=======================================================================
        # defaults
        #=======================================================================
        #get the current progress
        progress = self.progress() 
    
        #===================================================================
        # prechecks
        #===================================================================
        #make sure we have some slots connected
        """not sure how to do this"""
        
        #=======================================================================
        # reseting
        #=======================================================================
        if prog_raw is None:
            """
            would be nice to reset the progressBar.. .but that would be complicated
            """
            self.setProgress(0)
            return
        
        #=======================================================================
        # setting
        #=======================================================================
        if method=='append':
            prog = min(progress + prog_raw, 100)
        elif method=='raw':
            prog = prog_raw
        elif method == 'portion':
            rem_prog = 100-progress
            prog = progress + rem_prog*(prog_raw/100)
            
        assert prog<=100
        
        #===================================================================
        # emit signalling
        #===================================================================
        self.setProgress(prog)
        


    
    
if __name__ == '__main__':

    Qproj()
    

    
    print('finished')
    
    
    
    
    
    
    
    