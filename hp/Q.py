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
from qgis.gui import QgisInterface

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
from hp.dirz import get_valid_filename


#===============================================================================
# logging
#===============================================================================
mod_logger = logging.getLogger(__name__)


#===============================================================================
# funcs
#===============================================================================
class QAlgos(object):
    """
    common methods for applying algorthhims
    
    made a separate class just for organization
    """
    

    
    def __init__(self, 
                 **kwargs):
        
        super().__init__(**kwargs) #initilzie teh baseclass
        
        
    def _init_algos(self): #initiilize processing and add providers
        """
        crashing without raising an Exception
        """
    
    
        log = self.logger.getChild('_init_algos')
        
        if not isinstance(self.qap, QgsApplication):
            raise Error('qgis has not been properly initlized yet')
        
        from processing.core.Processing import Processing
    
        Processing.initialize() #crashing without raising an Exception
    
        QgsApplication.processingRegistry().addProvider(QgsNativeAlgorithms())
        
        assert not self.feedback is None, 'instance needs a feedback method for algos to work'
        
        log.info('processing initilzied w/ feedback: \'%s\''%(type(self.feedback).__name__))
        

        return True
    
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
    
    
    def selectbylocation(self, #select features (from main laye) by geoemtric relation with comp_vlay
                vlay, #vlay to select features from
                comp_vlay, #vlay to compare 
                
                result_type = 'select',
                
                method= 'new',  #Modify current selection by
                pred_l = ['intersect'],  #list of geometry predicate names
                
                selected_only = False, #selected features only on the comp_vlay
                
                #expectations
                allow_none = True,
                
                logger = None,

                ):
        
        #=======================================================================
        # setups and defaults
        #=======================================================================
        if logger is None: logger=self.logger    
        algo_nm = 'native:selectbylocation'   
        log = logger.getChild('selectbylocation')
        
        #===========================================================================
        # #set parameter translation dictoinaries
        #===========================================================================
        meth_d = {'new':0}
            
        pred_d = {
                'are within':6,
                'intersect':0,
                'overlap':5,
                  }
        
        #predicate (name to value)
        pred_l = [pred_d[pred_nm] for pred_nm in pred_l]
        
        if selected_only:
            intersect = self._get_sel_obj(comp_vlay)
        else:
            intersect = comp_vlay
    
        #=======================================================================
        # setup
        #=======================================================================
        ins_d = { 
            'INPUT' : vlay, 
            'INTERSECT' : intersect, 
            'METHOD' : meth_d[method], 
            'PREDICATE' : pred_l }
        
        log.debug('executing \'%s\' on \'%s\' with: \n     %s'
            %(algo_nm, vlay.name(), ins_d))
            
        #===========================================================================
        # #execute
        #===========================================================================
        _ = processing.run(algo_nm, ins_d,  feedback=self.feedback)
        
        
        #=======================================================================
        # check
        #=======================================================================
        fcnt = vlay.selectedFeatureCount()
        
        if fcnt == 0:
            msg = 'No features selected!'
            if allow_none:
                log.warning(msg)
            else:
                raise Error(msg)
            
        #=======================================================================
        # wrap
        #=======================================================================
        log.debug('selected %i (of %i) features from %s'
            %(vlay.selectedFeatureCount(),vlay.dataProvider().featureCount(), vlay.name()))
        
        return self._get_sel_res(vlay, result_type=result_type, logger=log, allow_none=allow_none)
        
    def saveselectedfeatures(self,#generate a memory layer from the current selection
                             vlay,
                             logger=None,
                             allow_none = False,
                             layname=None): 
        """
        TODO: add these intermediate layers to the store
        """
        
        
        #===========================================================================
        # setups and defaults
        #===========================================================================
        if logger is None: logger = self.logger
        log = logger.getChild('saveselectedfeatures')
        algo_nm = 'native:saveselectedfeatures'
        
        if layname is None: 
            layname = '%s_sel'%vlay.name()
              
        #=======================================================================
        # precheck
        #=======================================================================
        fcnt = vlay.selectedFeatureCount()
        if fcnt == 0:
            msg = 'No features selected!'
            if allow_none:
                log.warning(msg)
                return None
            else:
                raise Error(msg)
        
        log.debug('on \'%s\' with %i feats selected'%(
            vlay.name(), vlay.selectedFeatureCount()))
        #=======================================================================
        # # build inputs
        #=======================================================================
        ins_d = {'INPUT' : vlay,
                 'OUTPUT' : 'TEMPORARY_OUTPUT'}
        
        log.debug('\'native:saveselectedfeatures\' on \'%s\' with: \n   %s'
            %(vlay.name(), ins_d))
        
        #execute
        res_d = processing.run(algo_nm, ins_d,  feedback=self.feedback)

        
        res_vlay = res_d['OUTPUT']
        
        assert isinstance(res_vlay, QgsVectorLayer)
        #===========================================================================
        # wrap
        #===========================================================================

        res_vlay.setName(layname) #reset the name

        return res_vlay
    
    def cliprasterwithpolygon(self,
              rlay_raw,
              poly_vlay,
              layname = None,
              output = 'TEMPORARY_OUTPUT',
              result = 'layer', #type fo result to provide
                #layer: default, returns a raster layuer
                #fp: #returns the filepath result
              outResolution = None, #resultion for output. None = use input
              crsOut = None,
              logger = None,
                              ):
        """
        clipping a raster layer with a polygon mask using gdalwarp
        """
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger = self.logger
        log = logger.getChild('cliprasterwithpolygon')
        
        if layname is None:
            layname = '%s_clipd'%rlay_raw.name()
            
            
        algo_nm = 'gdal:cliprasterbymasklayer'
            

        #=======================================================================
        # precheck
        #=======================================================================
        assert isinstance(rlay_raw, QgsRasterLayer)
        assert isinstance(poly_vlay, QgsVectorLayer)
        assert 'Poly' in QgsWkbTypes().displayString(poly_vlay.wkbType())
        
        assert rlay_raw.crs() == poly_vlay.crs()
        
        #=======================================================================
        # cleanup outputs
        #=======================================================================
        if os.path.exists(output):
            assert self.overwrite
            os.remove(output) #gdal requires the file to be onge
            
        #=======================================================================
        # resolution
        #=======================================================================
        if not outResolution is None:
            assert isinstance(outResolution, int)
            setResolution = True
            
            log.debug('setting output resolution to %i'%outResolution)
        else:
            setResolution = False
            
            
        #=======================================================================
        # run algo        
        #=======================================================================

        
        ins_d = {   'ALPHA_BAND' : False,
                    'CROP_TO_CUTLINE' : True,
                    'DATA_TYPE' : 0,
                    'EXTRA' : '',
                    'INPUT' : rlay_raw,
                    'KEEP_RESOLUTION' : True, 
                    'MASK' : poly_vlay,
                    'MULTITHREADING' : False,
                    'NODATA' : None,
                    'OPTIONS' : '',
                    'OUTPUT' : output,
                    'SET_RESOLUTION' : setResolution,
                    'SOURCE_CRS' : None,
                    'TARGET_CRS' : crsOut,
                    'X_RESOLUTION' : outResolution,
                    'Y_RESOLUTION' : outResolution,
                     }
        
        log.debug('executing \'%s\' with ins_d: \n    %s \n\n'%(algo_nm, ins_d))
        
        res_d = processing.run(algo_nm, ins_d, feedback=self.feedback)
        
        log.debug('finished w/ \n    %s'%res_d)
        
        if not os.path.exists(res_d['OUTPUT']):
            """failing intermittently"""
            raise Error('failed to get a result')
        
        #=======================================================================
        # get the result
        #=======================================================================
        if result == 'layer':
            res_rlay = QgsRasterLayer(res_d['OUTPUT'], layname)
    
            #=======================================================================
            # #post check
            #=======================================================================
            assert isinstance(res_rlay, QgsRasterLayer), 'got bad type: %s'%type(res_rlay)
            assert res_rlay.isValid()
               
       
            res_rlay.setName(layname) #reset the name
               
            log.debug('finished w/ %s'%res_rlay.name())
          
            return res_rlay
        elif result == 'fp':
            return res_d['OUTPUT']
        else:
            raise Error('unrecognzied result kwarg: %s'%result)
        
        
    def warpreproject(self, #repojrect a raster
                              rlay_raw,
                              
                              crsOut = None, #crs to re-project to
                              layname = None,
                              options = 'COMPRESS=DEFLATE|PREDICTOR=2|ZLEVEL=9',
                              output = 'TEMPORARY_OUTPUT',
                              logger = None,
                              ):

        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger = self.logger
        log = logger.getChild('warpreproject')
        
        if layname is None:
            layname = '%s_rproj'%rlay_raw.name()
            
            
        algo_nm = 'gdal:warpreproject'
            
        if crsOut is None: crsOut = self.crs #just take the project's
        #=======================================================================
        # precheck
        #=======================================================================
        """the algo accepts 'None'... but not sure why we'd want to do this"""
        assert isinstance(crsOut, QgsCoordinateReferenceSystem), 'bad crs type'
        assert isinstance(rlay_raw, QgsRasterLayer)

        assert rlay_raw.crs() != crsOut, 'layer already on this CRS!'
            
            
        #=======================================================================
        # run algo        
        #=======================================================================

        
        ins_d =  {
             'DATA_TYPE' : 0,
             'EXTRA' : '',
             'INPUT' : rlay_raw,
             'MULTITHREADING' : False,
             'NODATA' : None,
             'OPTIONS' : options,
             'OUTPUT' : output,
             'RESAMPLING' : 0,
             'SOURCE_CRS' : None,
             'TARGET_CRS' : crsOut,
             'TARGET_EXTENT' : None,
             'TARGET_EXTENT_CRS' : None,
             'TARGET_RESOLUTION' : None,
          }
        
        log.debug('executing \'%s\' with ins_d: \n    %s \n\n'%(algo_nm, ins_d))
        
        res_d = processing.run(algo_nm, ins_d, feedback=self.feedback)
        
        log.debug('finished w/ \n    %s'%res_d)
        
        if not os.path.exists(res_d['OUTPUT']):
            """failing intermittently"""
            raise Error('failed to get a result')
        
        res_rlay = QgsRasterLayer(res_d['OUTPUT'], layname)

        #=======================================================================
        # #post check
        #=======================================================================
        assert isinstance(res_rlay, QgsRasterLayer), 'got bad type: %s'%type(res_rlay)
        assert res_rlay.isValid()
        assert rlay_raw.bandCount()==res_rlay.bandCount(), 'band count mismatch'
           
   
        res_rlay.setName(layname) #reset the name
           
        log.debug('finished w/ %s'%res_rlay.name())
          
        return res_rlay
    
    
    def _get_sel_obj(self, vlay): #get the processing object for algos with selections
        
        log = self.logger.getChild('_get_sel_obj')
        
        if vlay.selectedFeatureCount() == 0:
            raise Error('Nothing selected. exepects some pre selection')
        
        """consider moving this elsewhere"""
        #handle project layer store
        if QgsProject.instance().mapLayer(vlay.id()) is None:
            #layer not on project yet. add it
            if QgsProject.instance().addMapLayer(vlay, False) is None:
                raise Error('failed to add map layer \'%s\''%vlay.name())

        
       
        log.debug('based on %i selected features from \'%s\': %s'
                  %(len(vlay.selectedFeatureIds()), vlay.name(), vlay.selectedFeatureIds()))
            
        return QgsProcessingFeatureSourceDefinition(vlay.id(), True)
    
    
    def _get_sel_res(self, #handler for returning selection like results
                        vlay, #result layer (with selection on it
                         result_type='select',
                         
                         #expectiions
                         allow_none = False,
                         logger=None
                         ):
        
        #=======================================================================
        # setup
        #=======================================================================
        if logger is None: logger = self.logger
        log = logger.getChild('_get_sel_res')
        #=======================================================================
        # precheck
        #=======================================================================
        if vlay.selectedFeatureCount() == 0:
            if not allow_none:
                raise Error('nothing selected')
            
            return None

        
        #log.debug('user specified \'%s\' for result_type'%result_type)
        #=======================================================================
        # by handles
        #=======================================================================
        if result_type == 'select':
            #log.debug('user specified \'select\', doing nothing with %i selected'%vlay.selectedFeatureCount())
            
            result = None
            
        elif result_type == 'fids':
            
            result = vlay.selectedFeatureIds() #get teh selected feature ids
            
        elif result_type == 'feats':
            
            result =  {feat.id(): feat for feat in vlay.getSelectedFeatures()}
            
            
        elif result_type == 'layer':
            
            result = self.saveselectedfeatures(vlay, logger=log)
            
        else: 
            raise Error('unexpected result_type kwarg')
            
        return result
    
    
   
   
   
class Qproj(QAlgos, Basic):
    """
    common methods for Qgis projects
    """
    
    crsID_default = 'EPSG:4326'
    
    driverName = 'SpatiaLite' #default data creation driver type
    out_dName = driverName #default output driver/file type
    
    def __init__(self, 
                 feedback=None, 
                 crs = None,
                 crsID_default = None,
                 **kwargs):

        
        mod_logger.debug('Qproj super')
        
        super().__init__(**kwargs) #initilzie teh baseclass
        
        
        #=======================================================================
        # setup qgis
        #=======================================================================
        if feedback is None:
            feedback = MyFeedBackQ(logger=self.logger)
        self.feedback = feedback
        
        if not crsID_default is None: self.crsID_default=crsID_default
            
        self._init_qgis(crs=crs)
        

        self._init_algos()
        
        self._set_vdrivers()
        
        
        


        
        
        
        if not self.proj_checks():
            raise Error('failed checks')
        

        
        self.logger.info('Qproj __INIT__ finished w/ crs \'%s\''%self.qproj.crs().authid())
        
        
        return
    
    
    def _init_qgis(self, #instantiate qgis
                   crs=None,
                  gui = False): 
        """
        WARNING: need to hold this app somewhere. call in the module you're working in (scripts)
        
        """
        log = self.logger.getChild('_init_qgis')
        
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
            
        
        except:
            raise Error('QGIS failed to initiate')
        
        #=======================================================================
        # store the references
        #=======================================================================
        self.qap = app
        self.qproj = QgsProject.instance()
        self.mstore = QgsMapLayerStore() #build a new map store
        
        #=======================================================================
        # crs
        #=======================================================================
        if crs is None: 
            crs = QgsCoordinateReferenceSystem(self.crsID_default)
            
            
        assert isinstance(crs, QgsCoordinateReferenceSystem), 'bad crs type'
        assert crs.isValid()
            
        #self.crs = crs #just use the qproj
        self.qproj.setCrs(crs)
        
        log.info('set project crs to %s'%self.qproj.crs().authid())
        

    
    

    def _set_vdrivers(self):
        
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
        
        self.logger.info('built driver:extensions dict: \n    %s'%vlay_drivers)
        
        return
        
    def set_crs(self, #load, build, and set the project crs
                authid =  None):
        raise Error('depreciated')
        #=======================================================================
        # setup and defaults
        #=======================================================================
        log = self.logger.getChild('set_crs')
        
        if authid is None: 
            authid = self.crsID_default
        
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
    
    #===========================================================================
    # READ/WRITE-----
    #===========================================================================
    
    def vlay_write(self, #write  a VectorLayer
        vlay, out_fp, 

        driverName='GPKG',
        fileEncoding = "CP1250", 
        opts = QgsVectorFileWriter.SaveVectorOptions(), #empty options object
        overwrite=None,
        logger=mod_logger):
        """
        help(QgsVectorFileWriter.SaveVectorOptions)
        QgsVectorFileWriter.SaveVectorOptions.driverName='GPKG'
        
        
        opt2 = QgsVectorFileWriter.BoolOption(QgsVectorFileWriter.CreateOrOverwriteFile)
        
        help(QgsVectorFileWriter)
        

        
        """
        
        #==========================================================================
        # defaults
        #==========================================================================
        log = logger.getChild('vlay_write')
        if overwrite is None: overwrite=self.overwrite
        
    
        
        #===========================================================================
        # assemble options
        #===========================================================================
        opts.driverName = driverName
        opts.fileEncoding = fileEncoding
        
        
        #===========================================================================
        # checks
        #===========================================================================
        #file extension
        fhead, ext = os.path.splitext(out_fp)
        
        if not 'gpkg' in ext:
            raise Error('unexpected extension: %s'%ext)
        
        if os.path.exists(out_fp):
            msg = 'requested file path already exists!. overwrite=%s \n    %s'%(
                overwrite, out_fp)
            if overwrite:
                log.warning(msg)
                os.remove(out_fp) #workaround... should be away to overwrite with the QgsVectorFileWriter
            else:
                raise Error(msg)
            
        
        if vlay.dataProvider().featureCount() == 0:
            raise Error('\'%s\' has no features!'%(
                vlay.name()))
            
        if not vlay.isValid():
            Error('passed invalid layer')
            
            
            
            
        #=======================================================================
        # write
        #=======================================================================
        
        error = QgsVectorFileWriter.writeAsVectorFormatV2(
                vlay, out_fp, 
                QgsCoordinateTransformContext(),
                opts,
                )
        
        
        #=======================================================================
        # wrap and check
        #=======================================================================
          
        if error[0] == QgsVectorFileWriter.NoError:
            log.info('layer \' %s \' written to: \n     %s'%(vlay.name(),out_fp))
            return 
         
        raise Error('FAILURE on writing layer \' %s \'  with code:\n    %s \n    %s'%(vlay.name(),error, out_fp))
        
    
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
        assert not ext in ['tif'], 'passed invalid filetype: %s'%ext
        
        #=======================================================================
        # defaults
        #=======================================================================

        
        #===========================================================================
        # load the layer--------------
        #===========================================================================


        vlay_raw = QgsVectorLayer(file_path,fname,providerLib)


        #self.mstore.addMapLayer(vlay_raw)
            
            
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
        
        if not vlay_raw.crs() == self.qproj.crs():
            log.warning('crs: \'%s\' doesnt match project: %s'%(
                vlay_raw.crs().authid(), self.qproj.crs().authid()))
            
            
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
    
    def rlay_write(self, #make a local copy of the passed raster layer
                   rlayer, #raster layer to make a local copy of
                   extent = 'layer', #write extent control
                        #'layer': use the current extent (default)
                        #'mapCanvas': use the current map Canvas
                        #QgsRectangle: use passed extents
                   
                   
                   resolution = 'raw', #resolution for output
                   
                   
                   opts = ["COMPRESS=LZW"], #QgsRasterFileWriter.setCreateOptions
                   
                   out_dir = None, #directory for puts
                   newLayerName = None,
                   
                   logger=None,
                   ):
        """
        taken from CanFlood.hlpr.Q
        because  processing tools only work on local copies
        
        #=======================================================================
        # coordinate transformation
        #=======================================================================
        NO CONVERSION HERE!
        can't get native API to work. use gdal_warp instead
        """
        
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        if out_dir is None: out_dir = self.out_dir
        if newLayerName is None: newLayerName = rlayer.name()
        
        newFn = get_valid_filename('%s.tif'%newLayerName) #clean it
        
        out_fp = os.path.join(out_dir, newFn)

        
        log = logger.getChild('write_rlay')
        
        log.debug('on \'%s\' w/ \n    crs:%s \n    extents:%s\n    xUnits:%.4f'%(
            rlayer.name(), rlayer.crs(), rlayer.extent(), rlayer.rasterUnitsPerPixelX()))
        
        #=======================================================================
        # precheck
        #=======================================================================
        assert isinstance(rlayer, QgsRasterLayer)
        assert os.path.exists(out_dir)
        
        if os.path.exists(out_fp):
            msg = 'requested file already exists! and overwrite=%s \n    %s'%(
                self.overwrite, out_fp)
            if self.overwrite:
                log.warning(msg)
            else:
                raise Error(msg)

        #=======================================================================
        # extract info from layer
        #=======================================================================
        """consider loading the layer and duplicating the renderer?
        renderer = rlayer.renderer()"""
        provider = rlayer.dataProvider()
        
        
        #build projector
        projector = QgsRasterProjector()
        #projector.setCrs(provider.crs(), provider.crs())
        

        #build and configure pipe
        pipe = QgsRasterPipe()
        if not pipe.set(provider.clone()): #Insert a new known interface in default place
            raise Error("Cannot set pipe provider")
             
        if not pipe.insert(2, projector): #insert interface at specified index and connect
            raise Error("Cannot set pipe projector")

        #pipe = rlayer.pipe()
            
        
        #coordinate transformation
        """see note"""
        transformContext = self.qproj.transformContext()
        
        #=======================================================================
        # extents
        #=======================================================================
        if extent == 'layer':
            extent = rlayer.extent()
            
        elif extent=='mapCanvas':
            assert isinstance(self.iface, QgisInterface), 'bad key for StandAlone?'
            
            #get the extent, transformed to the current CRS
            extent =  QgsCoordinateTransform(
                self.qproj.crs(), 
                rlayer.crs(), 
                transformContext
                    ).transformBoundingBox(self.iface.mapCanvas().extent())
                
        assert isinstance(extent, QgsRectangle), 'expected extent=QgsRectangle. got \"%s\''%extent
        
        #expect the requested extent to be LESS THAN what we have in the raw raster
        assert rlayer.extent().width()>=extent.width(), 'passed extents too wide'
        assert rlayer.extent().height()>=extent.height(), 'passed extents too tall'
        #=======================================================================
        # resolution
        #=======================================================================
        #use the resolution of the raw file
        if resolution == 'raw':
            """this respects the calculated extents"""
            
            nRows = int(extent.height()/rlayer.rasterUnitsPerPixelY())
            nCols = int(extent.width()/rlayer.rasterUnitsPerPixelX())
            

            
        else:
            """dont think theres any decent API support for the GUI behavior"""
            raise Error('not implemented')

        

        
        #=======================================================================
        # #build file writer
        #=======================================================================

        file_writer = QgsRasterFileWriter(out_fp)
        #file_writer.Mode(1) #???
        
        if not opts is None:
            file_writer.setCreateOptions(opts)
        
        log.debug('writing to file w/ \n    %s'%(
            {'nCols':nCols, 'nRows':nRows, 'extent':extent, 'crs':rlayer.crs()}))
        
        #execute write
        error = file_writer.writeRaster( pipe, nCols, nRows, extent, rlayer.crs(), transformContext)
        
        log.info('wrote to file \n    %s'%out_fp)
        #=======================================================================
        # wrap
        #=======================================================================
        if not error == QgsRasterFileWriter.NoError:
            raise Error(error)
        
        assert os.path.exists(out_fp)
        
        assert QgsRasterLayer.isValidRasterFileName(out_fp),  \
            'requested file is not a valid raster file type: %s'%out_fp
        
        
        return out_fp
    
    
    
    def rlay_load(self, fp,  #load a raster layer and apply an aoi clip
                  aoi_vlay = None,
                  logger=None,
                  **clipKwargs):
        
        #=======================================================================
        # defautls
        #=======================================================================
        if logger is None: logger = self.logger
        log = logger.getChild('rlay_load')
        
        assert os.path.exists(fp), 'requested file does not exist: %s'%fp
        assert QgsRasterLayer.isValidRasterFileName(fp),  \
            'requested file is not a valid raster file type: %s'%fp
        
        basefn = os.path.splitext(os.path.split(fp)[1])[0]
        

        #Import a Raster Layer
        rlayer = QgsRasterLayer(fp, basefn)
        
        
        
        #===========================================================================
        # check
        #===========================================================================
        assert rlayer.isValid(), "Layer failed to load!"
        assert isinstance(rlayer, QgsRasterLayer), 'failed to get a QgsRasterLayer'
        
        if not rlayer.crs() == self.qproj.crs():
            log.warning('loaded layer \'%s\' crs mismatch!'%rlayer.name())

        log.debug('loaded \'%s\' from \n    %s'%(rlayer.name(), fp))
        
        #=======================================================================
        # aoi
        #=======================================================================
        rlay2 = rlayer
        if not aoi_vlay is None:
            self._check_aoi(aoi_vlay, logger=log)
            log.debug('clipping with %s'%aoi_vlay.name())
            assert isinstance(aoi_vlay, QgsVectorLayer)
            rlay2 = self.cliprasterwithpolygon(rlayer,aoi_vlay, 
                               logger=log, layname=rlayer.name(), **clipKwargs)
            
            #remove the raw
            mstore = QgsMapLayerStore() #build a new store
            mstore.addMapLayers([rlayer]) #add the layers to the store
            mstore.removeAllMapLayers() #remove all the layers

            
        
        return rlay2
    #===========================================================================
    # AOI methods--------
    #===========================================================================
    def load_aoi(self,
                 aoi_fp):
        
        log= self.logger.getChild('load_aoi')
        vlay = self.vlay_load(aoi_fp)
        self._check_aoi(vlay)
        self.aoi_vlay = vlay
        
        log.info('loaded aoi \'%s\''%vlay.name())
        self.aoi_vlay = vlay
        
        return vlay
    
    def slice_aoi(self, vlay,
                  aoi_vlay = None,
                  sfx = 'aoi',
                  logger=None):
        """
        also see misc.aoi_slicing
        """
        
        #=======================================================================
        # defaults
        #=======================================================================
        if aoi_vlay is None: aoi_vlay = self.aoi_vlay
        if logger is None: logger=self.logger
        log = logger.getChild('slice_aoi')
        
        #=======================================================================
        # prechecks
        #=======================================================================
        self._check_aoi(aoi_vlay)
        
        
        #=======================================================================
        # slice by aoi
        #=======================================================================

        vlay.removeSelection()
        log.info('slicing finv \'%s\' and %i feats w/ aoi \'%s\''%(
            vlay.name(),vlay.dataProvider().featureCount(), aoi_vlay.name()))
        

        
        res_vlay =  self.selectbylocation(vlay, aoi_vlay, result_type='layer', logger=log)
        
        #=======================================================================
        # wrap
        #=======================================================================
        vlay.removeSelection()
        #got some selection
        if isinstance(res_vlay, QgsVectorLayer):
            res_vlay.setName('%s_%s'%(vlay.name(), sfx))
            
        
        
            
        return res_vlay
    
    def _check_aoi(self, #special c hecks for AOI layers
                  vlay, 
                  logger=None):
        
        assert isinstance(vlay, QgsVectorLayer)
        assert 'Polygon' in QgsWkbTypes().displayString(vlay.wkbType())
        assert vlay.dataProvider().featureCount()==1
        assert vlay.crs() == self.qproj.crs(), 'aoi CRS (%s) does not match project (%s)'%(vlay.crs(), self.qproj.crs())
        
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
        

#===============================================================================
# standalone funcs--------
#===============================================================================

def vlay_get_fdf( #pull all the feature data and place into a df
                    vlay,
                    fmt='df', #result fomrat key. 
                        #dict: {fid:{fieldname:value}}
                        #df: index=fids, columns=fieldnames
                    
                    #limiters
                    request = None, #request to pull data. for more customized requestes.
                    fieldn_l = None, #or field name list. for generic requests
                    
                    #modifiers
                    reindex = None, #optinal field name to reindex df by
                    
                    #expectations
                    expect_all_real = False, #whether to expect all real results
                    allow_none = False,
                    
                    #db_f = False,
                    logger=mod_logger,
                    feedback=MyFeedBackQ()):
    """
    performance improvement
    
    Warning: requests with getFeatures arent working as expected for memory layers
    
    this could be combined with vlay_get_feats()
    also see vlay_get_fdata() (for a single column)
    
    CanFlood.hlpr.Q
    
    
    RETURNS
    a dictionary in the Qgis attribute dictionary format:
        key: generally feat.id()
        value: a dictionary of {field name: attribute value}

    """
    #===========================================================================
    # setups and defaults
    #===========================================================================
    log = logger.getChild('vlay_get_fdf')
    
    assert isinstance(vlay, QgsVectorLayer)
    all_fnl = [fieldn.name() for fieldn in vlay.fields().toList()]
    
    if fieldn_l is None: #use all the fields
        fieldn_l = all_fnl
        
   
    #field name check
    assert isinstance(fieldn_l, list)
    miss_l = set(fieldn_l).difference(all_fnl)
    assert len(miss_l)==0, '\'%s\' missing %i requested fields: %s'%(vlay.name(), len(miss_l), miss_l)
  

    
    if allow_none:
        if expect_all_real:
            raise Error('cant allow none and expect all reals')
        
    
    #===========================================================================
    # prechecks
    #===========================================================================
    if not reindex is None:
        if not reindex in fieldn_l:
            raise Error('requested reindexer \'%s\' is not a field name'%reindex)
    
    if not vlay.dataProvider().featureCount()>0:
        raise Error('no features!')

    if len(fieldn_l) == 0:
        raise Error('no fields!')
    
    if fmt=='dict' and not (len(fieldn_l)==len(all_fnl)):
        raise Error('dict results dont respect field slicing')
    
    assert hasattr(feedback, 'setProgress')
    
    #===========================================================================
    # build the request
    #===========================================================================
    feedback.setProgress(2)
    if request is None:
        """WARNING: this doesnt seem to be slicing the fields.
        see Alg().deletecolumns()
            but this will re-key things
        
        request = QgsFeatureRequest().setSubsetOfAttributes(fieldn_l,vlay.fields())"""
        request = QgsFeatureRequest()
        
    #never want geometry   
    request = request.setFlags(QgsFeatureRequest.NoGeometry) 
               

    log.debug('extracting data from \'%s\' on fields: %s'%(vlay.name(), fieldn_l))
    #===========================================================================
    # loop through each feature and extract the data
    #===========================================================================
    
    fid_attvs = dict() #{fid : {fieldn:value}}
    fcnt = vlay.dataProvider().featureCount()

    for indxr, feat in enumerate(vlay.getFeatures(request)):
        
        #zip values
        fid_attvs[feat.id()] = feat.attributes()
        
        feedback.setProgress((indxr/fcnt)*90)


    #===========================================================================
    # post checks
    #===========================================================================
    if not len(fid_attvs) == vlay.dataProvider().featureCount():
        log.debug('data result length does not match feature count')

        if not request.filterType()==3: #check if a filter fids was passed
            """todo: add check to see if the fiter request length matches tresult"""
            raise Error('no filter and data length mismatch')
        
    #check the field lengthes
    if not len(all_fnl) == len(feat.attributes()):
        raise Error('field length mismatch')

    #empty check 1
    if len(fid_attvs) == 0:
        log.warning('failed to get any data on layer \'%s\' with request'%vlay.name())
        if not allow_none:
            raise Error('no data found!')
        else:
            if fmt == 'dict': 
                return dict()
            elif  fmt == 'df':
                return pd.DataFrame()
            else:
                raise Error('unexpected fmt type')
            
    
    #===========================================================================
    # result formatting
    #===========================================================================
    log.debug('got %i data elements for \'%s\''%(
        len(fid_attvs), vlay.name()))
    
    if fmt == 'dict':
        
        return fid_attvs
    elif fmt=='df':
        
        #build the dict
        
        df_raw = pd.DataFrame.from_dict(fid_attvs, orient='index', columns=all_fnl)
        
        
        #handle column slicing and Qnulls
        """if the requester worked... we probably  wouldnt have to do this"""
        df = df_raw.loc[:, tuple(fieldn_l)].replace(NULL, np.nan)
        
        feedback.setProgress(95)
        
        if isinstance(reindex, str):
            """
            reindex='zid'
            view(df)
            """
            #try and add the index (fids) as a data column
            try:
                df = df.join(pd.Series(df.index,index=df.index, name='fid'))
            except:
                log.debug('failed to preserve the fids.. column already there?')
            
            #re-index by the passed key... should copy the fids over to 'index
            df = df.set_index(reindex, drop=True)
            
            log.debug('reindexed data by \'%s\''%reindex)
            
        return df
    
    else:
        raise Error('unrecognized fmt kwarg')

    
    

def view(#view the vector data (or just a df) as a html frame
        obj, logger=mod_logger,
        #**gfd_kwargs, #kwaqrgs to pass to vlay_get_fdatas() 'doesnt work well with the requester'
        ):
    
    if isinstance(obj, pd.DataFrame) or isinstance(obj, pd.Series):
        df = obj
    elif isinstance(obj, QgsVectorLayer):
        """this will index the viewed frame by fid"""
        df = vlay_get_fdf(obj)
    else:
        raise Error('got unexpected object type: %s'%type(obj))
    
    from hp.pd import view_web_df
    view_web_df(df)
    
    logger.info('viewer closed')
    
    return


def test_install(): #test your qgis install
    
    
    
    proj = Qproj()
    proj.get_install_info()
    
    
    
    
    
if __name__ == '__main__':

    test_install()
    

    
    print('finished')
    
    
    
    
    
    
    
    