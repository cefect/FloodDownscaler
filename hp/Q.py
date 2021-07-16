'''
Created on Oct. 8, 2020

@author: cefect
'''



#===============================================================================
# # standard imports -----------------------------------------------------------
#===============================================================================
import time, sys, os, logging, datetime, inspect, gc

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

#whitebox
#from processing_wbt.wbtprovider import WbtProvider 


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


#==============================================================================
# globals
#==============================================================================
fieldn_max_d = {'SpatiaLite':50, 'ESRI Shapefile':10, 'Memory storage':50, 'GPKG':50}

npc_pytype_d = {'?':bool,
                'b':int,
                'd':float,
                'e':float,
                'f':float,
                'q':int,
                'h':int,
                'l':int,
                'i':int,
                'g':float,
                'U':str,
                'B':int,
                'L':int,
                'Q':int,
                'H':int,
                'I':int, 
                'O':str, #this is the catchall 'object'
                }

type_qvar_py_d = {10:str, 2:int, 135:float, 6:float, 4:int, 1:bool, 16:datetime.datetime, 12:str} #QVariant.types to pythonic types

#parameters for lots of statistic algos
stat_pars_d = {'First': 0, 'Last': 1, 'Count': 2, 'Sum': 3, 'Mean': 4, 'Median': 5,
                'St dev (pop)': 6, 'Minimum': 7, 'Maximum': 8, 'Range': 9, 'Minority': 10,
                 'Majority': 11, 'Variety': 12, 'Q1': 13, 'Q3': 14, 'IQR': 15}

#spatial relation predicates
predicate_d = {'intersects':0,'contains':1,'equals':2,'touches':3,'overlaps':4,'within':5, 'crosses':6}


#===============================================================================
# funcs
#===============================================================================
class QAlgos(object):
    """
    common methods for applying algorthhims
    
    made a separate class just for organization
    """
    
    
    #projection operations
    """theres probably a nice way to get this from the users profile"""
    proj_d = {#{from:{to:operation}}
        'EPSG:4326':{
            'EPSG:3979':'+proj=pipeline +step +proj=unitconvert +xy_in=deg +xy_out=rad +step +proj=lcc +lat_0=49 +lon_0=-95 +lat_1=49 +lat_2=77 +x_0=0 +y_0=0 +ellps=GRS80',
            'EPSG:3857':'+proj=pipeline +step +proj=unitconvert +xy_in=deg +xy_out=rad +step +proj=webmerc +lat_0=0 +lon_0=0 +x_0=0 +y_0=0 +ellps=WGS84',
            'EPSG:2950':'+proj=pipeline +step +proj=unitconvert +xy_in=deg +xy_out=rad +step +proj=push +v_3 +step +proj=cart +ellps=WGS84 +step +inv +proj=helmert +x=-0.991 +y=1.9072 +z=0.5129 +rx=-0.0257899075194932 +ry=-0.0096500989602704 +rz=-0.0116599432323421 +s=0 +convention=coordinate_frame +step +inv +proj=cart +ellps=GRS80 +step +proj=pop +v_3 +step +proj=tmerc +lat_0=0 +lon_0=-73.5 +k=0.9999 +x_0=304800 +y_0=0 +ellps=GRS80',
            },
        'EPSG:3979':{
            'EPSG:3857':'+proj=pipeline +step +inv +proj=lcc +lat_0=49 +lon_0=-95 +lat_1=49 +lat_2=77 +x_0=0 +y_0=0 +ellps=GRS80 +step +proj=webmerc +lat_0=0 +lon_0=0 +x_0=0 +y_0=0 +ellps=WGS84',
            'EPSG:2950':'+proj=pipeline +step +inv +proj=lcc +lat_0=49 +lon_0=-95 +lat_1=49 +lat_2=77 +x_0=0 +y_0=0 +ellps=GRS80 +step +proj=tmerc +lat_0=0 +lon_0=-73.5 +k=0.9999 +x_0=304800 +y_0=0 +ellps=GRS80',
            },
        'EPSG:3402':{
            'EPSG:3857':'+proj=pipeline +step +inv +proj=tmerc +lat_0=0 +lon_0=-115 +k=0.9992 +x_0=500000 +y_0=0 +ellps=GRS80 +step +proj=webmerc +lat_0=0 +lon_0=0 +x_0=0 +y_0=0 +ellps=WGS84'
            },
        'EPSG:3857':{
            'EPSG:2950':'+proj=pipeline +step +inv +proj=webmerc +lat_0=0 +lon_0=0 +x_0=0 +y_0=0 +ellps=WGS84 +step +proj=push +v_3 +step +proj=cart +ellps=WGS84 +step +inv +proj=helmert +x=-0.991 +y=1.9072 +z=0.5129 +rx=-0.0257899075194932 +ry=-0.0096500989602704 +rz=-0.0116599432323421 +s=0 +convention=coordinate_frame +step +inv +proj=cart +ellps=GRS80 +step +proj=pop +v_3 +step +proj=tmerc +lat_0=0 +lon_0=-73.5 +k=0.9999 +x_0=304800 +y_0=0 +ellps=GRS80',
            },
        'EPSG:2950':{
            'EPSG:3979':'+proj=pipeline +step +inv +proj=tmerc +lat_0=0 +lon_0=-73.5 +k=0.9999 +x_0=304800 +y_0=0 +ellps=GRS80 +step +proj=lcc +lat_0=49 +lon_0=-95 +lat_1=49 +lat_2=77 +x_0=0 +y_0=0 +ellps=GRS80',
            }

        }
    
    #WARNING: some processing providers dont play well with high compression 
        #e.g. Whitebox doesnt recognize 'PREDICTOR' compression
    compress_d =  {
        'hiT':'COMPRESS=LERC_DEFLATE|PREDICTOR=2|ZLEVEL=9|MAX_Z_ERRROR=0.001', #nice for terrain
        'hi':'COMPRESS=DEFLATE|PREDICTOR=2|ZLEVEL=9',#Q default hi
        'med':'COMPRESS=LZW',
        'none':None        
        }

    
    def __init__(self, 
                 inher_d = {},
                 **kwargs):
        
        super().__init__(  #initilzie teh baseclassass
            inher_d = {**inher_d,
                **{'QAlgos':['context']}},
                        **kwargs) 
        
        
    def _init_algos(self,
                    context=None,
                    invalidGeometry=QgsFeatureRequest.GeometrySkipInvalid,
                        #GeometryNoCheck
                        #GeometryAbortOnInvalid
                        
                    ): #initiilize processing and add providers
        """
        crashing without raising an Exception
        """
    
    
        log = self.logger.getChild('_init_algos')
        
        if not isinstance(self.qap, QgsApplication):
            raise Error('qgis has not been properly initlized yet')
        
        #=======================================================================
        # build default co ntext
        #=======================================================================
        if context is None:

            context=QgsProcessingContext()
            context.setInvalidGeometryCheck(invalidGeometry)
            
        self.context=context
        
        #=======================================================================
        # init p[rocessing]
        #=======================================================================
        from processing.core.Processing import Processing

        
    
        Processing.initialize()  
    
        QgsApplication.processingRegistry().addProvider(QgsNativeAlgorithms())
        #QgsApplication.processingRegistry().addProvider(WbtProvider())
        
        #=======================================================================
        # #log all the agos
        # for alg in QgsApplication.processingRegistry().algorithms():
        #     log.debug("{}:{} --> {}".format(alg.provider().name(), alg.name(), alg.displayName()))
        #=======================================================================
        
        
        assert not self.feedback is None, 'instance needs a feedback method for algos to work'
        
        log.info('processing initilzied w/ feedback: \'%s\''%(type(self.feedback).__name__))
        

        return True
    
    
    #===========================================================================
    # NATIVE---------
    #===========================================================================
    

    
    def reproject(self,
                  vlay,
                  output='TEMPORARY_OUTPUT',
                  crsOut=None,
                  logger=None,
                  #layname=None,
                  selected_only=False,
                  ):
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log=logger.getChild('reproject')
        #if layname is None: layname=vlay.name()
        if crsOut is None: crsOut=self.qproj.crs()
        
        #=======================================================================
        # get operation
        #=======================================================================
        inid = vlay.crs().authid()
        outid = crsOut.authid()
        
        assert inid in self.proj_d, 'missing requested source crs: %s'%inid
        
        assert outid in self.proj_d[inid], 'missing requested op: %s to %s'%(inid, outid)
        
        #selection handling
        if selected_only:
            """not working well"""
            input_obj = self._get_sel_obj(vlay)
        else:
            input_obj = vlay
        
 
 
        #=======================================================================
        # execute
        #=======================================================================
        res_d = processing.run('native:reprojectlayer', 
                           { 'INPUT' : input_obj,
                             'OPERATION' : self.proj_d[inid][outid], 
                             'OUTPUT' : output,
                             'TARGET_CRS' : crsOut},  
                           feedback=self.feedback, context=self.context)


        log.info('finished  w/ %s'%res_d)
        return res_d
    
    def layerextent(self,
                    vlay,
                    output='TEMPORARY_OUTPUT',
                    precision=10, 
                    ):
        
        algo_nm = 'native:polygonfromlayerextent'
        
        ins_d = { 'INPUT' : vlay,'OUTPUT' : output, 'ROUND_TO' : precision }
        
        res_d = processing.run(algo_nm, ins_d, feedback=self.feedback)
        
        return res_d['OUTPUT']
    
    
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
        meth_d = {'new':0, 'subselection':2}
            
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
        _ = processing.run(algo_nm, ins_d,  feedback=self.feedback, context=self.context)
        
        
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
    
    def dissolve(self, #select features (from main laye) by geoemtric relation with comp_vlay
                vlay, #vlay to select features from
                fields = [], 
                output='TEMPORARY_OUTPUT',
                selected_only = False, #selected features only on the comp_vlay
                

                logger = None,

                ):
        
        #=======================================================================
        # setups and defaults
        #=======================================================================
        if logger is None: logger=self.logger    
        algo_nm = 'native:dissolve'   
        log = logger.getChild('dissolve')
        
        if selected_only:
            alg_input = self._get_sel_obj(vlay)
        else:
            alg_input = vlay

    
        #=======================================================================
        # setup
        #=======================================================================
        ins_d = { 'FIELD' : fields, 
                 'INPUT' : alg_input,
                 'OUTPUT' : output,
                 }
        
        #=======================================================================
        # log.debug('executing \'%s\' on \'%s\' with: \n     %s'
        #     %(algo_nm, vlay.name(), ins_d))
        #=======================================================================
            
        #===========================================================================
        # #execute
        #===========================================================================
        res_d = processing.run(algo_nm, ins_d,  feedback=self.feedback, context=self.context)
        

        return res_d['OUTPUT']
    

    
    def fixgeo(self, 
                vlay, #vlay to select features from

                output='TEMPORARY_OUTPUT',
                selected_only = False, #selected features only on the comp_vlay

                logger = None,

                ):
        
        #=======================================================================
        # setups and defaults
        #=======================================================================
        if logger is None: logger=self.logger    
        algo_nm = 'native:fixgeometries'   
        log = logger.getChild('fixgeo')
        
        if selected_only:
            alg_input = self._get_sel_obj(vlay)
        else:
            alg_input = vlay

    
        #=======================================================================
        # setup
        #=======================================================================
        ins_d = { 
                'INPUT' : alg_input,
                 'OUTPUT' : output,
                 }
        
        log.debug('executing \'%s\' on \'%s\' with: \n     %s'
            %(algo_nm, vlay, ins_d))
            
        #===========================================================================
        # #execute
        #===========================================================================
        res_d = processing.run(algo_nm, ins_d,  feedback=self.feedback)
        

        return res_d['OUTPUT']
    
    def centroids(self, 
                vlay, #vlay to select features from

                output='TEMPORARY_OUTPUT',
                selected_only = False, #selected features only on the comp_vlay

                logger = None,

                ):
        
        #=======================================================================
        # setups and defaults
        #=======================================================================
        if logger is None: logger=self.logger    
        algo_nm = 'native:centroids'   
        log = logger.getChild('centroids')
        
        if selected_only:
            alg_input = self._get_sel_obj(vlay)
        else:
            alg_input = vlay

    
        #=======================================================================
        # setup
        #=======================================================================
        ins_d = { 'ALL_PARTS' : False, 
                 'INPUT' : alg_input,
                  'OUTPUT' : output}
        
        log.debug('executing \'%s\' on \'%s\' with: \n     %s'
            %(algo_nm, vlay.name(), ins_d))
            
        #===========================================================================
        # #execute
        #===========================================================================
        res_d = processing.run(algo_nm, ins_d,  feedback=self.feedback)
        

        return res_d['OUTPUT']
    
    def pointonsurf(self, 
                vlay, #vlay to select features from

                output='TEMPORARY_OUTPUT',
                selected_only = False, #selected features only on the comp_vlay
                logger=None,

                ):
        
        #=======================================================================
        # setups and defaults
        #=======================================================================
        if logger is None: logger=self.logger    
        algo_nm = 'native:pointonsurface'   
        log = logger.getChild('pointonsurf')
        
        if selected_only:
            alg_input = self._get_sel_obj(vlay)
        else:
            alg_input = vlay

    
        #=======================================================================
        # setup
        #=======================================================================
        ins_d = { 'ALL_PARTS' : False, 
                 'INPUT' : alg_input,
                  'OUTPUT' : output}
        
        log.debug('executing \'%s\' on \'%s\' with: \n     %s'
            %(algo_nm, vlay.name(), ins_d))
            
        #===========================================================================
        # #execute
        #===========================================================================
        res_d = processing.run(algo_nm, ins_d,  feedback=self.feedback, context=self.context)
        

        return res_d['OUTPUT']
    
    def rastersampling(self, 
                vlay, #vlay with sampling features
                rlay, #raster to sample
                pfx='sample_',

                output='TEMPORARY_OUTPUT',
                selected_only = False, #selected features only on the comp_vlay
                logger=None,

                ):
        
        #=======================================================================
        # setups and defaults
        #=======================================================================
        if logger is None: logger=self.logger    
        algo_nm = 'native:rastersampling'   
        log = logger.getChild('rastersampling')
        
        if selected_only:
            alg_input = self._get_sel_obj(vlay)
        else:
            alg_input = vlay

    
        #=======================================================================
        # setup
        #=======================================================================
        ins_d = { 'COLUMN_PREFIX' : pfx, 
                 'INPUT' : alg_input,
                 'OUTPUT' : output, 
                 'RASTERCOPY' : rlay }
        
        log.debug('executing \'%s\' on \'%s\' with: \n     %s'
            %(algo_nm, vlay.name(), ins_d))
            
        #===========================================================================
        # #execute
        #===========================================================================
        res_d = processing.run(algo_nm, ins_d,  feedback=self.feedback, context=self.context)
        

        return res_d['OUTPUT']
    
    
    def saveselectedfeatures(self,#generate a memory layer from the current selection
                             vlay,
                             logger=None,
                             allow_none = False,
                             output='TEMPORARY_OUTPUT',
                             ): 
        """
        TODO: add these intermediate layers to the store
        """
        
        
        #===========================================================================
        # setups and defaults
        #===========================================================================
        if logger is None: logger = self.logger
        log = logger.getChild('saveselectedfeatures')
        algo_nm = 'native:saveselectedfeatures'
        
 
              
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
                 'OUTPUT' : output}
        
        log.debug('\'native:saveselectedfeatures\'  with: \n   %s'
            %(ins_d))
        
        #execute
        res_d = processing.run(algo_nm, ins_d,  feedback=self.feedback)

 

        return res_d['OUTPUT']
    
    def fillnodata(self,
                rlay,
                fval = 0, #value to fill nodata w/
                output='TEMPORARY_OUTPUT',
 
                logger=None,

                ):
        
        #=======================================================================
        # setups and defaults
        #=======================================================================
        if logger is None: logger=self.logger    
        algo_nm = 'native:fillnodata'   
        log = logger.getChild('fillnodata')
 
    
        #=======================================================================
        # setup
        #=======================================================================
        ins_d = { 'BAND' : 1, 
                 'FILL_VALUE' : fval,
                  'INPUT' : rlay,
                  'OUTPUT' : output}
        
        log.debug('executing \'%s\' on \'%s\' with: \n     %s'
            %(algo_nm, rlay, ins_d))
            
        #===========================================================================
        # #execute
        #===========================================================================
        res_d = processing.run(algo_nm, ins_d,  feedback=self.feedback, context=self.context)
        

        return res_d['OUTPUT']
    
    def mergevectorlayers(self,
                vlay_l,
                crs=None,
                output='TEMPORARY_OUTPUT',
                logger=None,

                ):
        
        #=======================================================================
        # setups and defaults
        #=======================================================================
        if logger is None: logger=self.logger    
        algo_nm = 'native:mergevectorlayers'  
        log = logger.getChild('mergevectorlayers')
 
        if crs is None: crs = self.qproj.crs()
        
        #=======================================================================
        # precheck
        #=======================================================================
        assert isinstance(vlay_l, list)
        #=======================================================================
        # setup
        #=======================================================================
        ins_d = { 'CRS' : crs, 'LAYERS' :vlay_l,      'OUTPUT' : output }
        
        log.debug('executing \'%s\' with: \n     %s'
            %(algo_nm,  ins_d))
            
        #===========================================================================
        # #execute
        #===========================================================================
        res_d = processing.run(algo_nm, ins_d,  feedback=self.feedback, context=self.context)
        

        return res_d['OUTPUT']
    
    def extractbyexpression(self,
                vlay,
                exp_str, #expression string to apply
                output='TEMPORARY_OUTPUT',
                fail_output = None, #how/if to output those failing the expression
                logger=None,

                ):
        
        #=======================================================================
        # setups and defaults
        #=======================================================================
        if logger is None: logger=self.logger    
        algo_nm = 'native:extractbyexpression'
        log = logger.getChild('extractbyexpression')
 
 
        #=======================================================================
        # setup
        #=======================================================================
        ins_d =    { 'EXPRESSION' : exp_str, 'INPUT' : vlay, 'OUTPUT' : output,
                    'FAIL_OUTPUT' : fail_output}
        
        log.debug('executing \'%s\' with: \n     %s'%(algo_nm,  ins_d))
            
 
        res_d = processing.run(algo_nm, ins_d,  feedback=self.feedback, context=self.context)
        

        return res_d
    
    def multiparttosingleparts(self,
            vlay,
            output='TEMPORARY_OUTPUT',
            logger=None,
            ):
        
        #=======================================================================
        # setups and defaults
        #=======================================================================
        if logger is None: logger=self.logger    
        algo_nm = 'native:multiparttosingleparts'
        log = logger.getChild('multiparttosingleparts')
 
        ins_d =    {'INPUT' : vlay, 'OUTPUT' : output}
        
        log.debug('executing \'%s\' with: \n     %s'%(algo_nm,  ins_d))
            
 
        res_d = processing.run(algo_nm, ins_d,  feedback=self.feedback, context=self.context)
        

        return res_d
    
    def clip(self,
            vlay,
            vlay_top,
            output='TEMPORARY_OUTPUT',
            logger=None,
            ):
        
        #=======================================================================
        # setups and defaults
        #=======================================================================
        if logger is None: logger=self.logger    
        algo_nm = 'native:clip'
        log = logger.getChild('clip')
 
        ins_d =    {'INPUT' : vlay, 'OUTPUT' : output, 'OVERLAY':vlay_top}
        
        log.debug('executing \'%s\' with: \n     %s'%(algo_nm,  ins_d))
            
 
        res_d = processing.run(algo_nm, ins_d,  feedback=self.feedback, context=self.context)
        

        return res_d
    

    #===========================================================================
    # QGIS--------
    #===========================================================================
    
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
    
    def joinattributesbylocation(self, 
         vlay, #base layer
         jvlay, #layer to join
         jvlay_fnl=[], #join layer field name list
         predicate='intersects', 
         prefix='',
         method=0, #join type
             # 0: Create separate feature for each matching feature (one-to-many)
             #1: Take attributes of the first matching feature only (one-to-one)
             #2: Take attributes of the feature with largest overlap only (one-to-one)
        output='TEMPORARY_OUTPUT',
             
        logger=None,
                                 ):
        """
        also see canflood.hlpr.Q for more sophisticated version
        
        dropped all the data checks and warnings here
        """
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log = logger.getChild('joinattributesbylocation')
        
        algo_nm = 'qgis:joinattributesbylocation'
        
        #=======================================================================
        # assemble parameters
        #=======================================================================
        assert predicate in predicate_d, 'unrecognized predicarte: %s' %predicate

        
        
        pars_d = { 'DISCARD_NONMATCHING' : False, 
                  'INPUT' : vlay, 
                  'JOIN' : jvlay, 
                  'JOIN_FIELDS' : jvlay_fnl, 
                  'METHOD' : method, 
                  'NON_MATCHING' : 'TEMPORARY_OUTPUT', 
                  'OUTPUT' : output, 
                  'PREDICATE' : [predicate_d[predicate]], #only accepting single predicate
                  'PREFIX' : prefix }
        

        #=======================================================================
        # execute
        #=======================================================================
        log.debug('%s w/ \n%s'%(algo_nm, pars_d))
        res_d = processing.run(algo_nm, pars_d, feedback=self.feedback)
        
        """just leaving the output as is
        #retriieve results
        if os.path.exists(output):
            res_vlay = self.vlay_load(output)
        else:
            res_vlay = res_d[output]
            
        assert isinstance(res_vlay, QgsVectorLayer)
        """
            
        result = res_d['OUTPUT']
        
        join_cnt  = res_d['JOINED_COUNT']
        
        vlay_nomatch = res_d['NON_MATCHING'] #Unjoinable features from first layer
        
        #=======================================================================
        # warp
        #=======================================================================
        ofcnt = vlay.dataProvider().featureCount()
        jfcnt = jvlay.dataProvider().featureCount()
        miss_cnt = ofcnt-join_cnt
        
        if not miss_cnt>=0:
            log.warning('got negative miss_cnt: %i'%miss_cnt)
            """this can happen when a base feature intersects multiple join features for method=0"""
        
        log.info('finished joining \'%s\' (%i feats) to \'%s\' (%i feats)\n    %i hits and %i misses'%(
            vlay.name(), ofcnt, jvlay.name(), jfcnt, join_cnt, miss_cnt))
        
        return result, miss_cnt
    
    def joinbylocationsummary(self,
            vlay, #layer to add stats to
             join_vlay, #layer to extract stats from
             jlay_fieldn_l, #list of field names to extract from the join_vlay
             selected_only=False, #limit to selected only on the main
             jvlay_selected_only = False, #only consider selected features on the join layer

             predicate_l = ['intersects'],#list of geometric serach predicates
             smry_l = ['sum'], #data summaries to apply
             discard_nomatch = False, #Discard records which could not be joined
             
             use_raw_fn=False, #whether to convert names back to the originals
             layname=None,
                                 
                     ):
        """
        WARNING: This ressets the fids
        
        discard_nomatch: 
            TRUE: two resulting layers have no features in common
            FALSE: in layer retains all non matchers, out layer only has the non-matchers?
        
        """
        
        """
        view(join_vlay)
        """
        #=======================================================================
        # presets
        #=======================================================================
        algo_nm = 'qgis:joinbylocationsummary'
        
        predicate_d = {'intersects':0,'contains':1,'equals':2,'touches':3,'overlaps':4,'within':5, 'crosses':6}
        summaries_d = {'count':0, 'unique':1, 'min':2, 'max':3, 'range':4, 'sum':5, 'mean':6}

        log = self.logger.getChild('joinbylocationsummary')
        
        #=======================================================================
        # defaults
        #=======================================================================
        if isinstance(jlay_fieldn_l, set):
            jlay_fieldn_l = list(jlay_fieldn_l)
            
            
        #convert predicate to code
        pred_code_l = [predicate_d[pred_name] for pred_name in predicate_l]
            
        #convert summaries to code
        sum_code_l = [summaries_d[smry_str] for smry_str in smry_l]
        
        
        if layname is None:  layname = '%s_jsmry'%vlay.name()
            
        #=======================================================================
        # prechecks
        #=======================================================================
        if not isinstance(jlay_fieldn_l, list):
            raise Error('expected a list')
        
        #check requested join fields
        fn_l = [f.name() for f in join_vlay.fields()]
        s = set(jlay_fieldn_l).difference(fn_l)
        assert len(s)==0, 'requested join fields not on layer: %s'%s
        
        #check crs
        assert join_vlay.crs().authid() == vlay.crs().authid()
                
        #=======================================================================
        # set selection
        #=======================================================================
        if selected_only:
            main_input = self._get_sel_obj(vlay)
        else:
            main_input=vlay

        if jvlay_selected_only:
            join_input = self._get_sel_obj(join_vlay)
        else:
            join_input = join_vlay

        #=======================================================================
        # #assemble pars
        #=======================================================================
        ins_d = { 'DISCARD_NONMATCHING' : discard_nomatch,
                  'INPUT' : main_input,
                   'JOIN' : join_input,
                   'JOIN_FIELDS' : jlay_fieldn_l,
                  'OUTPUT' : 'TEMPORARY_OUTPUT', 
                  'PREDICATE' : pred_code_l, 
                  'SUMMARIES' : sum_code_l,
                   }
        
        log.debug('executing \'%s\' with ins_d: \n    %s'%(algo_nm, ins_d))
 
        res_d = processing.run(algo_nm, ins_d, feedback=self.feedback, context=self.context)
 
        res_vlay = res_d['OUTPUT']
 
        #===========================================================================
        # post formatting
        #===========================================================================
        res_vlay.setName(layname) #reset the name
        
        #get new field names
        nfn_l = set([f.name() for f in res_vlay.fields()]).difference([f.name() for f in vlay.fields()])
        
        """
        view(res_vlay)
        """
        #=======================================================================
        # post check
        #=======================================================================
        #=======================================================================
        # for fn in nfn_l:
        #     rser = vlay_get_fdata(res_vlay, fn)
        #     if rser.isna().all().all():
        #         log.warning('%s \'%s\' got all nulls'%(vlay.name(), fn))
        #=======================================================================

        
        #=======================================================================
        # rename fields
        #=======================================================================
        if use_raw_fn:
            raise Error('?')
            #===================================================================
            # assert len(smry_l)==1, 'rename only allowed for single sample stat'
            # rnm_d = {s:s.replace('_%s'%smry_l[0],'') for s in nfn_l}
            # 
            # s = set(rnm_d.values()).symmetric_difference(jlay_fieldn_l)
            # assert len(s)==0, 'failed to convert field names'
            # 
            # res_vlay = vlay_rename_fields(res_vlay, rnm_d, logger=log)
            # 
            # nfn_l = jlay_fieldn_l
            #===================================================================
        
        
        
        log.info('sampled \'%s\' w/ \'%s\' (%i hits) and \'%s\'to get %i new fields \n    %s'%(
            join_vlay.name(), vlay.name(), res_vlay.dataProvider().featureCount(), 
            smry_l, len(nfn_l), nfn_l))
        
        return res_vlay, nfn_l

    
    #===========================================================================
    # GDAL---------
    #===========================================================================

    
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
              options = [],
              dataType=0, # 0: Use Input Layer Data Type
                #6: Float32
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
                    'DATA_TYPE' : dataType,
                    'EXTRA' : '',
                    'INPUT' : rlay_raw,
                    'KEEP_RESOLUTION' : True, 
                    'MASK' : poly_vlay,
                    'MULTITHREADING' : True,
                    'NODATA' : -9999,
                    'OPTIONS' : options,
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
        return self._get_rlay_res(res_d, result, layname=layname)
    
    def extrapNoData(self,
                     rlay,
                     dist, #maximum pixes to search for interpolation values
                     iterations=0,
                     output='TEMPORARY_OUTPUT',
                     options='',
                     logger=None,
                     ):
        if logger is None: logger=self.logger
        log = logger.getChild('extrapNoData')
        
        algo_nm = 'gdal:fillnodata'
        
        ins_d = { 'BAND' : 1, 
                 'DISTANCE' : dist, 'EXTRA' : '',
          'INPUT' : rlay,
           'ITERATIONS' : iterations, 
           'MASK_LAYER' : None, 'NO_MASK' : False,
          'OPTIONS' : options, 
          'OUTPUT' : output }
 
        log.info('dist=%.2f on %s'%(dist, rlay))
        return processing.run(algo_nm, ins_d, feedback=self.feedback, context=self.context)

        
        
    def warpreproject(self, #repojrect a raster
                              rlay_raw,
                              
                              crsOut = None, #crs to re-project to
                              layname = None,
                              compression = 'none',
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
             'OPTIONS' : self.compress_d[compression],
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
    
    def mergeraster(self, #merge a set of raster layers
                  rlays_l,
                  crsOut = None, #crs to re-project to
                  layname = None,
                  compression = 'hiT',
                  output = 'TEMPORARY_OUTPUT',
                  logger = None,
                              ):

        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger = self.logger
        log = logger.getChild('mergeraster')
        
        if layname is None:
            layname = 'merge'
            
            
        algo_nm = 'gdal:merge'
            
        if crsOut is None: crsOut = self.qproj.crs() #just take the project's
        #=======================================================================
        # precheck
        #=======================================================================
        """the algo accepts 'None'... but not sure why we'd want to do this"""
        assert isinstance(crsOut, QgsCoordinateReferenceSystem), 'bad crs type'
        assert isinstance(rlays_l, list)
        assert (output == 'TEMPORARY_OUTPUT') or (output.endswith('.tif')) 
        
 
        first, bc = True, None
        for r in rlays_l: 
            if not os.path.exists(r):
                assert isinstance(r, QgsRasterLayer)
                assert r.crs() != crsOut, 'layer already on this CRS!'
                
                if first:
                    first = False
                else:
                    assert r.bandCount() == bc
                bc = r.bandCount()
        
        #=======================================================================
        # execute
        #=======================================================================
        ins_d = { 'DATA_TYPE' : 5, 
                 'EXTRA' : '',
                  'INPUT' : rlays_l, 
                  #'NODATA_INPUT' : -9999, 
                  'NODATA_OUTPUT' : -9999,
                   'OPTIONS' : self.compress_d[compression],
                  'OUTPUT' : output, 
                  'PCT' : False, 'SEPARATE' : False }
        
        log.debug('executing \'%s\' with ins_d: \n    %s \n\n'%(algo_nm, ins_d))
        
        res_d = processing.run(algo_nm, ins_d, feedback=self.feedback, context=self.context)
        
        #=======================================================================
        # wrap
        #=======================================================================
        log.debug('finished w/ \n    %s'%res_d)
        
        if not os.path.exists(res_d['OUTPUT']):
            """failing intermittently"""
            raise Error('failed to get a result')
        
        
        if output == 'TEMPORARY_OUTPUT':
            res_rlay = QgsRasterLayer(res_d['OUTPUT'], layname)
            

            assert isinstance(res_rlay, QgsRasterLayer), 'got bad type: %s'%type(res_rlay)
            assert res_rlay.isValid()
            assert bc==res_rlay.bandCount(), 'band count mismatch'
               
       
            res_rlay.setName(layname) #reset the name
               
            log.debug('finished w/ %s'%res_rlay.name())
        else:
            res_rlay = res_d['OUTPUT']
          
        return res_rlay
        
        
    

    
    def rasterize_value(self, #build a rastser with a fixed value from a polygon
                bval, #fixed value to burn,
                poly_vlay, #polygon layer with geometry
                resolution=10,
                output = 'TEMPORARY_OUTPUT',
                result = 'layer', #type fo result to provide
                #layer: default, returns a raster layuer
                #fp: #returns the filepath result
                layname=None,
                logger=None,
                  ):
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log = logger.getChild('rasterize')
        if layname is None: layname = '%s_%.2f'%(poly_vlay.name(), bval)
        algo_nm = 'gdal:rasterize'
        
        
        
        """
        extents =  QgsRectangle(-127.6, 44.1, -106.5, 54.1)
        """
        #=======================================================================
        # get extents
        #=======================================================================
        rect = poly_vlay.extent()
        
        extent = '%s,%s,%s,%s'%(rect.xMinimum(), rect.xMaximum(), rect.yMinimum(), rect.yMaximum())+ \
                ' [%s]'%poly_vlay.crs().authid()

        
        #=======================================================================
        # build pars
        #=======================================================================
        pars_d = { 'BURN' : bval, #fixed value to burn
                  'EXTENT' : extent,
                   #'EXTENT' : '1221974.375000000,1224554.125000000,466981.406300000,469354.031300000 [EPSG:3005]',
                    'EXTRA' : '', 'FIELD' : '', 
                    'HEIGHT' : resolution, 
                    'WIDTH' : resolution, 
                    'UNITS' : 1,  #Georeferenced units 
                    'INIT' : None, #Pre-initialize the output image with value
                     
                      'INVERT' : False,
                   'NODATA' : -9999, 'DATA_TYPE' : 5,'OPTIONS' : '',
                   'INPUT' : poly_vlay, 'OUTPUT' : output,
                    
                     
                      }
        
        log.debug('%s w/ \n    %s'%(algo_nm, pars_d))
        res_d = processing.run(algo_nm, pars_d, feedback=self.feedback)
        
        #laod teh rlay
        
    
        return self._get_rlay_res(res_d, result, layname=layname)
    
    #===========================================================================
    # WHITEBOX------
    #===========================================================================
    def BreachDepressionsLeastCost(self,
                                   rlay,
                                   dist=100,
                                   output='TEMPORARY_OUTPUT',
                                   ):
        raise Error('cant get the whitebox provider to work')
        
        ins_d = { 'dem' : rlay,
                  'dist' : dist, 
                  'fill' : True, 'flat_increment' : None, 'max_cost' : None, 
                  'min_dist' : True,
                   'output' : output }
        
        
        
        algo_nm='wbt:BreachDepressionsLeastCost'
        
        res_d = processing.run(algo_nm, ins_d, feedback=self.feedback)
        
        return res_d
    #===========================================================================
    # helpers-------
    #===========================================================================
    
    def _get_sel_obj(self, vlay): #get the processing object for algos with selections
        
        log = self.logger.getChild('_get_sel_obj')
        
        if vlay.selectedFeatureCount() == 0:
            raise Error('Nothing selected on \'%s\'. exepects some pre selection'%(vlay.name()))
        
        #=======================================================================
        # """consider moving this elsewhere"""
        # #handle project layer store
        # if QgsProject.instance().mapLayer(vlay.id()) is None:
        #     #layer not on project yet. add it
        #     if QgsProject.instance().addMapLayer(vlay, False) is None:
        #         raise Error('failed to add map layer \'%s\''%vlay.name())
        #=======================================================================
            
        
        #handle project layer store
        if self.qproj.mapLayer(vlay.id()) is None:
            #layer not on project yet. add it
            if self.qproj.addMapLayer(vlay, False) is None:
                raise Error('failed to add map layer \'%s\''%vlay.name())
            
 
            
       
        log.debug('based on %i selected features from \'%s\''%(len(vlay.selectedFeatureIds()), vlay.name()))
        
        return QgsProcessingFeatureSourceDefinition(source=vlay.id(), 
                                                    selectedFeaturesOnly=True, 
                                                    featureLimit=-1, 
                                                    geometryCheck=QgsFeatureRequest.GeometryAbortOnInvalid)
        
 
        #return QgsProcessingFeatureSourceDefinition(vlay.id(), True)
    
    
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
    
    def _get_rlay_res(self, res_d, result, layname=None):
        
        if result == 'layer':
            res_rlay = QgsRasterLayer(res_d['OUTPUT'], layname)
    
            #=======================================================================
            # #post check
            #=======================================================================
            assert isinstance(res_rlay, QgsRasterLayer), 'got bad type: %s'%type(res_rlay)
            assert res_rlay.isValid()
               
       
            res_rlay.setName(layname) #reset the name
               

          
            return res_rlay
        elif result == 'fp':
            return res_d['OUTPUT']
        else:
            raise Error('unrecognzied result kwarg: %s'%result)
    

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
                 
                 #aois
                 aoi_fp = None,
                 aoi_vlay = None,
                 
                 #inheritance
                 session=None, #parent session for child mode
                 inher_d = {},
                 **kwargs):

        
        mod_logger.debug('Qproj super')
        
        super().__init__(
            inher_d = {**inher_d,
                **{'Qproj':['qap', 'qproj', 'vlay_drivers']}},
            
            **kwargs) #initilzie teh baseclass
        

        
        #=======================================================================
        # setup qgis
        #=======================================================================
        if feedback is None:
            feedback = MyFeedBackQ(logger=self.logger)
        self.feedback = feedback
        
        if not crsID_default is None: self.crsID_default=crsID_default
            
        #standalone
        if session is None:
            self._init_qgis(crs=crs)
            
            self._init_algos()
            
            self._set_vdrivers()
            
        #child mode
        else:
            self.inherit(session=session)
            """having a separate mstore is one of the main reasons to use children"""
            self.mstore = QgsMapLayerStore() #build a new map store 
        
        
        if not self.proj_checks():
            raise Error('failed checks')
        """
        self.tag
        """
        #=======================================================================
        # aois
        #=======================================================================
        if not aoi_fp is None:
            self.load_aoi(aoi_fp)
        
        if not aoi_vlay is None:
            assert aoi_fp is None, 'cant pass a layer and a filepath'
            self._check_aoi(aoi_vlay)
            self.aoi_vlay= aoi_vlay
        
        self.logger.info('Qproj __INIT__ finished w/ crs \'%s\''%self.qproj.crs().authid())
        
        
        return
    
    
    def _init_qgis(self, #instantiate qgis
                   crs=None,
                  gui = False): 
        """
        WARNING: need to hold this app somewhere. call in the module you're working in (scripts)
        
        """
        log = self.logger.getChild('_init_qgis')
        
        
        #=======================================================================
        # init the application
        #=======================================================================
        
        try:
            
            QgsApplication.setPrefixPath(r'C:/OSGeo4W64/apps/qgis-ltr', True)
            
            app = QgsApplication([], gui)

            app.initQgis()

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
                  addSpatialIndex=True,
                  dropZ=True,
                  reproj=False, #whether to reproject hte layer to match the project
                  
                  set_proj_crs = False, #set the project crs from this layer

                  logger = None,
                  ):
        
        #=======================================================================
        # setup
        #=======================================================================
        if logger is None: logger=self.logger
        log = logger.getChild('vlay_load')
        
        
        #file
        assert os.path.exists(file_path), file_path
        fname, ext = os.path.splitext(os.path.split(file_path)[1])
        assert not ext in ['tif'], 'passed invalid filetype: %s'%ext
        log.info('on %s'%file_path)
        mstore = QgsMapLayerStore() 
        #===========================================================================
        # load the layer--------------
        #===========================================================================

        vlay_raw = QgsVectorLayer(file_path,fname,providerLib)
  
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
 
        #=======================================================================
        # clean------
        #=======================================================================
        #spatial index
        if addSpatialIndex and (not vlay_raw.hasSpatialIndex()==QgsFeatureSource.SpatialIndexPresent):
            self.createspatialindex(vlay_raw, logger=log)
            


        #=======================================================================
        # #zvalues
        #=======================================================================
        if dropZ:
            vlay1 = processing.run('native:dropmzvalues', 
                                   {'INPUT':vlay_raw, 'OUTPUT':'TEMPORARY_OUTPUT', 'DROP_Z_VALUES':True},  
                                   feedback=self.feedback, context=self.context)['OUTPUT']
            mstore.addMapLayer(vlay_raw)
        else:
            vlay1 = vlay_raw
            
        #=======================================================================
        # #CRS
        #=======================================================================
        if not vlay1.crs() == self.qproj.crs():
            log.warning('\'%s\' crs: \'%s\' doesnt match project: %s. reproj=%s. set_proj_crs=%s'%(
                vlay_raw.name(), vlay_raw.crs().authid(), self.qproj.crs().authid(), reproj, set_proj_crs))
            
            if reproj:
                vlay2 = self.reproject(vlay1, logger=log)['OUTPUT']
                mstore.addMapLayer(vlay1)

                
            elif set_proj_crs:
                self.qproj.setCrs(vlay1.crs())
                vlay2 = vlay1
                
            else:
                vlay2 = vlay1
                
        else:
            vlay2 = vlay1
 
                
        vlay2.setName(vlay_raw.name())
 
        #=======================================================================
        # wrap
        #=======================================================================
        #add to project
        'needed for some algorhithims?. moved to algos'
        #vlay = self.qproj.addMapLayer(vlay, False)
        

        dp = vlay2.dataProvider()
                
        log.info('loaded \'%s\' as \'%s\' \'%s\'  with %i feats crs: \'%s\' from file: \n     %s'
                    %(vlay2.name(), dp.storageType(), 
                      QgsWkbTypes().displayString(vlay2.wkbType()), 
                      dp.featureCount(), 
                      vlay2.crs().authid(),
                      file_path))
        
        mstore.removeAllMapLayers()
        
        return vlay2
    
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
        # coordinate transformation
        #=======================================================================
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
        error = file_writer.writeRaster( pipe, nCols, nRows, extent, rlayer.crs(), transformContext,
                                         feedback=RasterFeedback(log), #not passing any feedback
                                         )
        
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
                  
                  #crs handling
                  set_proj_crs = False, #set the project crs from this layer
                  reproj=False,
                  **clipKwargs):
        
        #=======================================================================
        # defautls
        #=======================================================================
        if logger is None: logger = self.logger
        log = logger.getChild('rlay_load')
        mstore = QgsMapLayerStore() #build a new store
        
        assert os.path.exists(fp), 'requested file does not exist: %s'%fp
        assert QgsRasterLayer.isValidRasterFileName(fp),  \
            'requested file is not a valid raster file type: %s'%fp
        
        basefn = os.path.splitext(os.path.split(fp)[1])[0]
        
        #=======================================================================
        # load
        #=======================================================================
        rlay_raw = QgsRasterLayer(fp, basefn)
        
        #===========================================================================
        # check
        #===========================================================================
        
        assert isinstance(rlay_raw, QgsRasterLayer), 'failed to get a QgsRasterLayer'
        assert rlay_raw.isValid(), "Layer failed to load!"
        log.debug('loaded \'%s\' from \n    %s'%(rlay_raw.name(), fp))
        
        #=======================================================================
        # #CRS
        #=======================================================================
        if not rlay_raw.crs() == self.qproj.crs():
            log.warning('\'%s\'  match fail (%s v %s) \n    reproj=%s set_proj_crs=%s'%(
                rlay_raw.name(), rlay_raw.crs().authid(), self.qproj.crs().authid(), reproj, set_proj_crs))
            
            if reproj:
                raise Error('not implemented')

                mstore.addMapLayer(rlay_raw)

                
            elif set_proj_crs:
                self.qproj.setCrs(rlay_raw.crs())
                rlay1 = rlay_raw
                
            else:
                rlay1 = rlay_raw
                
        else:
            rlay1 = rlay_raw
        
        #=======================================================================
        # aoi
        #=======================================================================

        if not aoi_vlay is None:

            log.debug('clipping with %s'%aoi_vlay.name())

            rlay2 = self.cliprasterwithpolygon(rlay1,aoi_vlay, 
                               logger=log, layname=rlay1.name(), **clipKwargs)
            
            #remove the raw
            
            mstore.addMapLayer(rlay1) #add the layers to the store
            
        else:
            rlay2 = rlay1

        #=======================================================================
        # wrap
        #=======================================================================
        mstore.removeAllMapLayers() #remove all the layers
            
        
        return rlay2
    #===========================================================================
    # AOI methods--------
    #===========================================================================
    def load_aoi(self,
                 aoi_fp, **kwargs):
        
        log= self.logger.getChild('load_aoi')
        vlay = self.vlay_load(aoi_fp, **kwargs)
        self._check_aoi(vlay)
        self.aoi_vlay = vlay
        
        log.info('loaded aoi \'%s\''%vlay.name())

        
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
    
    #===========================================================================
    # vlay methods-------------
    #===========================================================================
    def vlay_new_df(self, #build a vlay from a df
            df_raw,
            
            geo_d = None, #container of geometry objects {fid: QgsGeometry}

            crs=None,
            gkey = None, #data field linking with geo_d (if None.. uses df index)
            
            layname='df',
            
            index = False, #whether to include the index as a field
            logger=None, 

            ):
        """
        performance enhancement over vlay_new_df
            simpler, clearer
            although less versatile
        """
        #=======================================================================
        # setup
        #=======================================================================
        if crs is None: crs = self.qproj.crs()
        if logger is None: logger = self.logger
            
        log = logger.getChild('vlay_new_df')
        
        
        #=======================================================================
        # index fix
        #=======================================================================
        df = df_raw.copy()
        
        if index:
            if not df.index.name is None:
                coln = df.index.name
                df.index.name = None
            else:
                coln = 'index'
                
            df[coln] = df.index
            
        #=======================================================================
        # precheck
        #=======================================================================
        
        
        #make sure none of hte field names execeed the driver limitations
        max_len = fieldn_max_d[self.driverName]
        
        #check lengths
        boolcol = df_raw.columns.str.len() >= max_len
        
        if np.any(boolcol):
            log.warning('passed %i columns which exeed the max length=%i for driver \'%s\'.. truncating: \n    %s'%(
                boolcol.sum(), max_len, self.driverName, df_raw.columns[boolcol].tolist()))
            
            
            df.columns = df.columns.str.slice(start=0, stop=max_len-1)

        
        #make sure the columns are unique
        assert df.columns.is_unique
        
        #check the geometry
        if not geo_d is None:
            assert isinstance(geo_d, dict)
            if not gkey is None:
                assert gkey in df_raw.columns
        
                #assert 'int' in df_raw[gkey].dtype.name
                
                #check gkey match
                l = set(df_raw[gkey].drop_duplicates()).difference(geo_d.keys())
                assert len(l)==0, 'missing %i \'%s\' keys in geo_d: %s'%(len(l), gkey, l)
                
            #against index
            else:
                
                #check gkey match
                l = set(df_raw.index).difference(geo_d.keys())
                assert len(l)==0, 'missing %i (of %i) fid keys in geo_d: %s'%(len(l), len(df_raw), l)

        #===========================================================================
        # assemble the fields
        #===========================================================================
        #column name and python type
        fields_d = {coln:np_to_pytype(col.dtype) for coln, col in df.items()}
        
        #fields container
        qfields = fields_build_new(fields_d = fields_d, logger=log)
        
        #=======================================================================
        # assemble the features
        #=======================================================================
        #convert form of data
        
        feats_d = dict()
        for fid, row in df.iterrows():
    
            feat = QgsFeature(qfields, fid) 
            
            #loop and add data
            for fieldn, value in row.items():
    
                #skip null values
                if pd.isnull(value): continue
                
                #get the index for this field
                findx = feat.fieldNameIndex(fieldn) 
                
                #get the qfield
                qfield = feat.fields().at(findx)
                
                #make the type match
                ndata = qtype_to_pytype(value, qfield.type(), logger=log)
                
                #set the attribute
                if not feat.setAttribute(findx, ndata):
                    raise Error('failed to setAttribute')
                
            #setgeometry
            if not geo_d is None:
                if gkey is None:
                    gobj = geo_d[fid]
                else:
                    gobj = geo_d[row[gkey]]
                
                feat.setGeometry(gobj)
            
            #stor eit
            feats_d[fid]=feat
        
        log.debug('built %i \'%s\'  features'%(
            len(feats_d),
            QgsWkbTypes.geometryDisplayString(feat.geometry().type()),
            ))
        
        
        #=======================================================================
        # get the geo type
        #=======================================================================\
        if not geo_d is None:
            gtype = QgsWkbTypes().displayString(next(iter(geo_d.values())).wkbType())
        else:
            gtype='None'

            
            
        #===========================================================================
        # buidl the new layer
        #===========================================================================
        vlay = vlay_new_mlay(gtype,
                             crs, 
                             layname,
                             qfields,
                             list(feats_d.values()),
                             logger=log,
                             )
        self.createspatialindex(vlay, logger=log)
        #=======================================================================
        # post check
        #=======================================================================
        if not geo_d is None:
            if vlay.wkbType() == 100:
                raise Error('constructed layer has NoGeometry')



        
        return vlay
    
    def __exit__(self, #destructor
                 *args,**kwargs):
        
        #clear your map store
        self.mstore.removeAllMapLayers()
        #print('clearing mstore')
        
        super().__exit__(*args,**kwargs) #initilzie teh baseclass
        
    


    
    

    
class RasterFeedback(QgsRasterBlockFeedback):
    prog=0
    
    def __init__(self,log):
        
        self.logger=log
        
        super().__init__()
        
        
    def onNewData(self):
        print('onNewData')
        
    def setProgress(self, newProg):
        self.prog+=newProg
        print(self.prog)
 
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
                    selected=False, #for only get selected features
                    
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
        
    #get selected only
    if selected: 
        request.setFilterFids(vlay.selectedFeatureIds())
        assert vlay.selectedFeatureCount()>0, 'passed selected=True but nothing is selected!'
        
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
    
    
def vlay_get_fdata(vlay, 
                   fieldn,  #get data from a field
                   request=None,
                   selected=False,
                   ):
    
    #===========================================================================
    # precheck
    #===========================================================================
    fnl = [f.name() for f in vlay.fields().toList()]
    assert fieldn in fnl, 'requested field \'%s\' not on layer'%fieldn
    
    #===========================================================================
    # build request
    #===========================================================================
    if request is None:
        request = QgsFeatureRequest()
        
    request = request.setFlags(QgsFeatureRequest.NoGeometry)
    request = request.setSubsetOfAttributes([fieldn],vlay.fields())
    
    
    
    
    if selected:
        """
        todo: check if there is already a fid filter placed on the reuqester
        """
 
        sfids = vlay.selectedFeatureIds()
        
        request = request.setFilterFids(sfids)
        
    return {f.id():f.attribute(fieldn) for f in vlay.getFeatures(request)}
    

def vlay_get_geo( #get geometry dict from layer
        vlay,
        request = None, #additional requester (limiting fids). fieldn still required. additional flags added
        selected = False,
        logger=mod_logger,
        ):
    
    log = logger.getChild('vlay_get_geo')
    
    #===========================================================================
    # build the request
    #===========================================================================
    
    if request is None:
        request = QgsFeatureRequest()
    request = request.setNoAttributes() #dont get any attributes
    
    
    if selected:
        """
        todo: check if there is already a fid filter placed on the reuqester
        """
        log.debug('limiting data pull to %i selected features on \'%s\''%(
            vlay.selectedFeatureCount(), vlay.name()))
        
        sfids = vlay.selectedFeatureIds()
        
        request = request.setFilterFids(sfids)
        
        
    #===========================================================================
    # loop through and collect hte data
    #===========================================================================

    d = {f.id():f.geometry() for f in vlay.getFeatures(request)}
    
    log.debug('retrieved %i attributes from features on \'%s\''%(
        len(d), vlay.name()))
    
    #===========================================================================
    # checks
    #===========================================================================
    assert len(d)>0
    
    #===========================================================================
    # wrap
    #===========================================================================
    return d

def vlay_new_mlay(#create a new mlay
                      gtype, #"Point", "LineString", "Polygon", "MultiPoint", "MultiLineString", or "MultiPolygon".
                      crs,
                      layname,
                      qfields,
                      feats_l,

                      logger=mod_logger,
                      ):
        #=======================================================================
        # defaults
        #=======================================================================
        log = logger.getChild('vlay_new_mlay')

        #=======================================================================
        # prechecks
        #=======================================================================
        if not isinstance(layname, str):
            raise Error('expected a string for layname, isntead got %s'%type(layname))
        
        if gtype=='None':
            log.warning('constructing mlay w/ \'None\' type')
        #=======================================================================
        # assemble into new layer
        #=======================================================================
        #initilzie the layer
        EPSG_code=int(crs.authid().split(":")[1]) #get teh coordinate reference system of input_layer
        uri = gtype+'?crs=epsg:'+str(EPSG_code)+'&index=yes'
        
        vlaym = QgsVectorLayer(uri, layname, "memory")
        
        # add fields
        if not vlaym.dataProvider().addAttributes(qfields):
            raise Error('failed to add fields')
        
        vlaym.updateFields()
        
        #add feats
        if not vlaym.dataProvider().addFeatures(feats_l):
            raise Error('failed to addFeatures')
        
        vlaym.updateExtents()
        

        
        #=======================================================================
        # checks
        #=======================================================================
        if vlaym.wkbType() == 100:
            msg = 'constructed layer \'%s\' has NoGeometry'%vlaym.name()
            if gtype == 'None':
                log.debug(msg)
            else:
                raise Error(msg)

        
        log.debug('constructed \'%s\''%vlaym.name())
        return vlaym
    
def field_new(fname, 
              pytype=str, 
              driverName = 'SpatiaLite', #desired driver (to check for field name length limitations)
              fname_trunc = True, #whether to truncate field names tha texceed the limit
              logger=mod_logger): #build a QgsField
    
    #===========================================================================
    # precheck
    #===========================================================================
    if not isinstance(fname, str):
        raise IOError('expected string for fname')
    
    #vector layer field name lim itation
    max_len = fieldn_max_d[driverName]
    """
    fname = 'somereallylongname'
    """
    if len(fname) >max_len:
        log = logger.getChild('field_new')
        log.warning('got %i (>%i)characters for passed field name \'%s\'. truncating'%(len(fname), max_len, fname))
        
        if fname_trunc:
            fname = fname[:max_len]
        else:
            raise Error('field name too long')
        
    
    qtype = ptype_to_qtype(pytype)
    
    """
    #check this type
    QMetaType.typeName(QgsField(fname, qtype).type())
    
    QVariant.String
    QVariant.Int
    
     QMetaType.typeName(new_qfield.type())
    
    """
    #new_qfield = QgsField(fname, qtype)
    new_qfield = QgsField(fname, qtype, typeName=QMetaType.typeName(QgsField(fname, qtype).type()))
    
    return new_qfield

def fields_build_new( #build qfields from different data containers
                    samp_d = None, #sample data from which to build qfields {fname: value}
                    fields_d = None, #direct data from which to build qfields {fname: pytype}
                    fields_l = None, #list of QgsField objects
                    logger=mod_logger):

    log = logger.getChild('fields_build_new')
    #===========================================================================
    # buidl the fields_d
    #===========================================================================
    if (fields_d is None) and (fields_l is None): #only if we have nothign better to start with
        if samp_d is None: 
            log.error('got no data to build fields on!')
            raise IOError
        
        fields_l = []
        for fname, value in samp_d.items():
            if pd.isnull(value):
                log.error('for field \'%s\' got null value')
                raise IOError
            
            elif inspect.isclass(value):
                raise IOError
            
            fields_l.append(field_new(fname, pytype=type(value)))
            
        log.debug('built %i fields from sample data'%len(fields_l))
        
    
    
    #===========================================================================
    # buidl the fields set
    #===========================================================================
    elif fields_l is None:
        fields_l = []
        for fname, ftype in fields_d.items():
            fields_l.append(field_new(fname, pytype=ftype))
            
        log.debug('built %i fields from explicit name/type'%len(fields_l))
            
        #check it 
        if not len(fields_l) == len(fields_d):
            raise Error('length mismatch')
            
    elif fields_d is None: #check we have the other
        raise IOError
    
    
    
            
    #===========================================================================
    # build the Qfields
    #===========================================================================
    
    Qfields = QgsFields()
    
    fail_msg_d = dict()
    
    for indx, field in enumerate(fields_l): 
        if not Qfields.append(field):
            fail_msg_d[indx] = ('%i failed to append field \'%s\''%(indx, field.name()), field)

    #report
    if len(fail_msg_d)>0:
        for indx, (msg, field) in fail_msg_d.items():
            log.error(msg)
            
        raise Error('failed to write %i fields'%len(fail_msg_d))
    
    """
    field.name()
    field.constraints().constraintDescription()
    field.length()
    
    """
    
    
    #check it 
    if not len(Qfields) == len(fields_l):
        raise Error('length mismatch')


    return Qfields
    

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

#==============================================================================
# type conversions----------------
#==============================================================================

def np_to_pytype(npdobj, logger=mod_logger):
    
    if not isinstance(npdobj, np.dtype):
        raise Error('not passed a numpy type')
    
    try:
        return npc_pytype_d[npdobj.char]

    except Exception as e:
        log = logger.getChild('np_to_pytype')
        
        if not npdobj.char in npc_pytype_d.keys():
            log.error('passed npdtype \'%s\' not found in the conversion dictionary'%npdobj.name)
            
        raise Error('failed oto convert w/ \n    %s'%e)
    

def qtype_to_pytype( #convert object to the pythonic type taht matches the passed qtype code
        obj, 
        qtype_code, #qtupe code (qfield.type())
        logger=mod_logger): 
    
    if is_qtype_match(obj, qtype_code): #no conversion needed
        return obj 
    
    
    #===========================================================================
    # shortcut for nulls
    #===========================================================================
    if qisnull(obj):
        return None

        
    
        
    
    
    #get pythonic type for this code
    py_type = type_qvar_py_d[qtype_code]
    
    try:
        return py_type(obj)
    except:
        #datetime
        if qtype_code == 16:
            return obj.toPyDateTime()
        
        
        log = logger.getChild('qtype_to_pytype')
        if obj is None:
            log.error('got NONE type')
            
        elif isinstance(obj, QVariant):
            log.error('got a Qvariant object')
            
        else:
            log.error('unable to map object \'%s\' of type \'%s\' to type \'%s\''
                      %(obj, type(obj), py_type))
            
            
            """
            QMetaType.typeName(obj)
            """
        raise IOError
    
def ptype_to_qtype(py_type, logger=mod_logger): #get the qtype corresponding to the passed pytype
    """useful for buildign Qt objects
    
    really, this is a reverse 
    
    py_type=str
    
    """
    if not inspect.isclass(py_type):
        logger.error('got unexpected type \'%s\''%type(py_type))
        raise Error('bad type')
    
    #build a QVariant object from this python type class, then return its type
    try:
        qv = QVariant(py_type())
    except:
        logger.error('failed to build QVariant from \'%s\''%type(py_type))
        raise IOError
    
    """
    #get the type
    QMetaType.typeName(qv.type())
    """
    
    
    return qv.type()

#==============================================================================
# type checks-----------------
#==============================================================================

def qisnull(obj):
    if obj is None:
        return True
    
    if isinstance(obj, QVariant):
        if obj.isNull():
            return True
        else:
            return False
        
    
    if pd.isnull(obj):
        return True
    else:
        return False
    
def is_qtype_match(obj, qtype_code, logger=mod_logger): #check if the object matches the qtype code
    log = logger.getChild('is_qtype_match')
    
    #get pythonic type for this code
    try:
        py_type = type_qvar_py_d[qtype_code]
    except:

        if not qtype_code in type_qvar_py_d.keys():
            log.error('passed qtype_code \'%s\' not in dict from \'%s\''%(qtype_code, type(obj)))
            raise IOError
    
    if not isinstance(obj, py_type):
        #log.debug('passed object of type \'%s\' does not match Qvariant.type \'%s\''%(type(obj), QMetaType.typeName(qtype_code)))
        return False
    else:
        return True

def test_install(): #test your qgis install
    
    
    
    proj = Qproj()
    proj.get_install_info()
    
    
    
    
    
if __name__ == '__main__':

    test_install()
    

    
    print('finished')
    
    
    
    
    
    
    
    