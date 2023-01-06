'''
Created on Jan. 6, 2023

@author: cefect

validating downscaling results
'''
import logging, os
import numpy as np
import pandas as pd
import rasterio as rio
from sklearn.metrics import confusion_matrix

from hp.rio import (
    RioSession, assert_rlay_simple, get_stats, assert_spatial_equal,
    )

from hp.logr import get_new_console_logger
from fdsc.scripts.scripts import Master_Session


def rlay_extract(fp,
                 window=None, masked=False,
 
                 ):
    
    """load rlay data and arrays"""
    with rio.open(fp, mode='r') as ds:
        assert_rlay_simple(ds)
        stats_d = get_stats(ds) 
 
        ar = ds.read(1, window=window, masked=masked)
        
    return stats_d, ar 

class Valid_Session(Master_Session):
    pass


class ValidateRaster(RioSession):
    confusion_ser=None
    
    """compute validation metrics for a raster by comparing to some true raster""" 
    def __init__(self,
                 true_fp=None,
                 pred_fp=None,
                 sample_pts_fp=None, 
                 logger=None,
                 **kwargs):
        
        #=======================================================================
        # pre init
        #=======================================================================
        if logger is None:
            logger=get_new_console_logger(level=logging.DEBUG)
        
        
        super().__init__(logger=logger,**kwargs)
                
        #=======================================================================
        # load rasters
        #=======================================================================
        """using ocnditional loading mostly for testing"""
        
        if not true_fp is None:
            #compare rasters
            if not pred_fp is None:
                assert_spatial_equal(true_fp, pred_fp)
                
            stats_d, self.true_ar = rlay_extract(true_fp)
            self.logger.info('loaded true raster from file w/\n    %s'%stats_d)
            

            
        if not pred_fp is None:
            stats_d, self.pred_ar = rlay_extract(pred_fp)
            self.logger.info('loaded pred raster from file w/\n    %s'%stats_d)
            
        #=======================================================================
        # attachments
        #=======================================================================
 
 
    
    #===========================================================================
    # grid inundation metrics------
    #===========================================================================
    def get_hitRate(self,**kwargs):
        """proportion of wet benchmark data that was replicated by the model"""
        #log, true_ar, pred_ar = self._func_setup('hitRate', **kwargs)
        
        cf_ser = self._confusion(**kwargs)
        
        return cf_ser['TP']/(cf_ser['TP']+cf_ser['FN'])
    
    def get_falseAlarms(self,**kwargs):
        #log, true_ar, pred_ar = self._func_setup('hitRate', **kwargs)
        
        cf_ser = self._confusion(**kwargs)
        
        return cf_ser['FP']/(cf_ser['TP']+cf_ser['FP'])
    
    def get_criticalSuccessIndex(self, **kwargs):
        """critical success index. accounts for both overprediction and underprediction"""
        
        cf_ser = self._confusion(**kwargs)
        
        return cf_ser['TP']/(cf_ser['TP']+cf_ser['FP']+cf_ser['FN'])
    
    
    def get_errorBias(self, **kwargs):
        """indicates whether the model has a tendency toward overprediction or underprediction"""
        
        cf_ser = self._confusion(**kwargs)
        return cf_ser['FP']/cf_ser['FN']
    
    def get_inundation_all(self, **kwargs):
        """convenience for getting all the inundation metrics
        using notation from Wing (2017)
        """
        
        d= {
            'H':self.get_hitRate(**kwargs),
            'F':self.get_falseAlarms(**kwargs),
            'C':self.get_criticalSuccessIndex(**kwargs),
            'E':self.get_errorBias(**kwargs),
            }
        
        log, _, _ = self._func_setup('inun', **kwargs)
        log.info('computed all inundation metrics:\n    %s'%d)
        return d
        
        
        
        
 
        
 
        
    #===========================================================================
    # private helpers
    #===========================================================================
    def _confusion(self, **kwargs):
        """retrieve or construct the wet/dry confusion series"""
        if self.confusion_ser is None:
            log, true_ar, pred_ar = self._func_setup('hitRate', **kwargs)
            
            #convert to boolean (true=wet=nonnull)
            true_arB, pred_arB = np.invert(np.isnan(true_ar)), np.invert(np.isnan(pred_ar))
            
            assert_partial_wet(true_arB)
            assert_partial_wet(pred_arB)
            
            """
            true_arB.sum()
            """
 
     
            
            #fancy labelling
            self.confusion_ser = pd.Series(confusion_matrix(true_arB.ravel(), pred_arB.ravel(),
                                                            labels=[False, True]).ravel(),
                      index = ['TN', 'FP', 'FN', 'TP'])
            
            log.info('generated wet-dry confusion matrix on %s\n    %s'%(
                str(true_ar.shape), self.confusion_ser.to_dict()))
            
        return self.confusion_ser.copy()

    def _func_setup(self, dkey, 
                    logger=None,  
                    true_ar=None, pred_ar=None,
                    ):
        """common function default setup
        
       
 
        
        """
 
        if logger is None:
            logger = self.logger
        log = logger.getChild(dkey)
        
        if true_ar is None: true_ar=self.true_ar
        if pred_ar is None: pred_ar=self.pred_ar
            
        return log, true_ar, pred_ar
    

def assert_partial_wet(ar):
    """assert a boolean array has some trues and some falses (but not all)"""
    if not __debug__: # true if Python was not started with an -O option
        return
    
    assert ar.dtype == np.dtype('bool')
    
    if np.all(ar):
        raise AssertionError('all true')
    if np.all(np.invert(ar)):
        raise AssertionError('all false')
    
    
    
