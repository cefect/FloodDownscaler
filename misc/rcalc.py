'''
Created on Jun. 4, 2021

@author: cefect

raster calculator
'''


#===============================================================================
# imports-----------
#===============================================================================
import os, datetime
import pandas as pd
import numpy as np

start =  datetime.datetime.now()
print('start at %s'%start)
today_str = datetime.datetime.today().strftime('%Y%m%d')


from hp.exceptions import Error
from hp.dirz import force_open_dir
 
from hp.Q import Qproj, QgsRasterLayer #only for Qgis sessions
 

from qgis.analysis import QgsRasterCalculatorEntry, QgsRasterCalculator


class Session(Qproj):
    
 
    
    
    def __init__(self,
 
                  **kwargs):
        
        super().__init__( **kwargs)
        
    def run_calc(self, #build 2 mask from a HAND layer with upper/lower buffers
                      hand_rlay, #hand raster
                      formula,
                      layname='result',
                     logger=None, 
                      ):
        """EXAZMPLE from InsCrve.hybrid"""
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log=logger.getChild('mask')
 
        if isinstance(hand_rlay, str):
            hand_rlay = QgsRasterLayer(hand_rlay, os.path.basename(hand_rlay).replace('.tif', ''))
            
        log.info("on %s w/ %s"%(hand_rlay.name(), formula))
        
        #=======================================================================
        # build calculator constructors
        #=======================================================================
        rcentry = QgsRasterCalculatorEntry()
        rcentry.raster=hand_rlay
        rcentry.ref = '%s@1'%hand_rlay.name()
        rcentry.bandNumber=1
        
 
        #=======================================================================
        # assemble parameters
        #=======================================================================
 
        outputFile = os.path.join(self.out_dir,layname+'.tif' )
        outputExtent  = hand_rlay.extent()
        outputFormat = 'GTiff'
        nOutputColumns = hand_rlay.width()
        nOutputRows = hand_rlay.height()
        rasterEntries =[rcentry]
        
        #=======================================================================
        # execute
        #=======================================================================
        """throwing depreciation warning"""
        rcalc = QgsRasterCalculator(formula.format(rcentry.ref), outputFile, outputFormat, outputExtent,
                            nOutputColumns, nOutputRows, rasterEntries)
        
        result = rcalc.processCalculation(feedback=self.feedback)
        
        #=======================================================================
        # check    
        #=======================================================================
        if not result == 0:
            raise Error(rcalc.lastError())
        
        assert os.path.exists(outputFile)
        
        
        log.info('saved result to: \n    %s'%outputFile)
            
        #=======================================================================
        # retrieve result
        #=======================================================================
        
        
        return outputFile
    
    
    def get_hand_mask(self, #build a mask from a HAND layer with upper/lower buffers
                      hand_rlay, #hand raster
                      hval, #median value
                     lowb= -2, #lower buffer to apply to  hval
                     hib = 2,
                     logger=None
                      ):
        
        
        """EXAZMPLE from InsCrve.hybrid"""
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log=logger.getChild('mask')
        lo, hi = max(self.hval_min, hval+lowb), hval+hib
        log.info("on %s w/ %.2f and %.2f"%(hand_rlay.name(), lo, hi))
        
        #=======================================================================
        # build calculator constructors
        #=======================================================================
        rcentry = QgsRasterCalculatorEntry()
        rcentry.raster=hand_rlay
        rcentry.ref = '%s@1'%hand_rlay.name()
        rcentry.bandNumber=1
 
        #=======================================================================
        # assemble parameters
        #=======================================================================
 
        formula = '{0}*(({0}>{1})/({0}>{1}))*(({0}<{2})/({0}<{2}))'.format(rcentry.ref, lo, hi)
        layname = '%s_mask_%.2f_%.2f'%(hand_rlay.name(), lo, hi)
        outputFile = os.path.join(self.out_dir,layname+'.tif' )
        outputExtent  = hand_rlay.extent()
        outputFormat = 'GTiff'
        nOutputColumns = hand_rlay.width()
        nOutputRows = hand_rlay.height()
        rasterEntries =[rcentry]
        
        #=======================================================================
        # execute
        #=======================================================================
        """throwing depreciation warning"""
        rcalc = QgsRasterCalculator(formula, outputFile, outputFormat, outputExtent,
                            nOutputColumns, nOutputRows, rasterEntries)
        
        result = rcalc.processCalculation(feedback=self.feedback)
        
        #=======================================================================
        # check    
        #=======================================================================
        if not result == 0:
            raise Error(rcalc.lastError())
        
        assert os.path.exists(outputFile)
        
        
        log.info('saved result to: \n    %s'%outputFile)
            
        #=======================================================================
        # retrieve result
        #=======================================================================
        rlay = QgsRasterLayer(outputFile, layname)
        
        return rlay